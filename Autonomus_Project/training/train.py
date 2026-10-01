import os
# Force compatibility with Keras 2 to avoid internal TensorFlow errors
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
from dataclasses import replace

from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.system import multiprocessing as tf_multiprocessing
from tf_agents.utils import common

# Import our custom modules
from envs.tf_wrapper import TFStagHuntWrapper, ParallelStagHuntEnv
from agents.networks import ActorNetwork, CriticNetwork
from agents.ppo_custom import CustomPPO, MultiAgentPPO
from training.config import TrainingConfig
from training.memory import MemoryManager
from training.rollouts import collect_rollout, train_ppo_epochs
from training.scheduling import EntropyScheduler
from training.visualization import generate_annotated_heatmap

_PARALLEL_MP_INITIALIZED = False
ENVIRONMENT_IDS = {
    'hunt': 'StagHunt-Hunt-v0',
    'harvest': 'StagHunt-Harvest-v0',
    'escalation': 'StagHunt-Escalation-v0',
}


def _find_convergence_iteration(training_metrics, config, reward_key='return'):
    """Return the first 1-based iteration with a stable good reward rate."""
    steps_per_rollout = max(config.rollout_steps - 1, 1)
    rewards = np.asarray(
        [metric[reward_key] / steps_per_rollout for metric in training_metrics],
        dtype=float,
    )
    window = config.convergence_window
    if len(rewards) < window:
        return None

    for end_index in range(window, len(rewards) + 1):
        reward_window = rewards[end_index - window:end_index]
        if (
            np.all(reward_window >= config.convergence_reward_rate)
            and np.std(reward_window) <= config.convergence_max_rate_std
        ):
            return end_index
    return None


def _annealed_learning_rate(config, iteration):
    progress = min(iteration / max(config.num_iterations - 1, 1), 1.0)
    return config.learning_rate + progress * (config.learning_rate_end - config.learning_rate)


def _build_experiment_id(config: TrainingConfig, requested_name=None):
    config_payload = json.dumps(
        {key: value for key, value in vars(config).items() if key != 'seed'},
        sort_keys=True,
        separators=(',', ':')
    )
    config_hash = hashlib.sha1(config_payload.encode('utf-8')).hexdigest()[:8]
    if requested_name:
        return f"{requested_name}_{config_hash}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"experiment_{timestamp}_{config_hash}"


def _create_experiment_paths(config: TrainingConfig, experiment_name=None, root_dir=None):
    experiment_id = _build_experiment_id(config, experiment_name)
    experiment_dir = Path(root_dir) if root_dir is not None else Path('./experiments') / experiment_id
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_dir}. "
            "Choose a different --experiment-name."
        )

    paths = {
        'experiment_id': experiment_id,
        'root': experiment_dir,
        'checkpoints': experiment_dir / 'checkpoints',
        'training_logs': experiment_dir / 'training_logs',
    }
    for key, path in paths.items():
        if key != 'experiment_id':
            path.mkdir(parents=True, exist_ok=True)

    metadata = {
        'experiment_id': experiment_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'hyperparameters': vars(config),
    }
    (experiment_dir / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
    (experiment_dir / 'README.md').write_text(
        f"# Experiment `{experiment_id}`\n\n"
        "This directory contains the complete outputs for one isolated training run.\n\n"
        "- Hyperparameters: `metadata.json`\n"
        "- Checkpoints: `checkpoints/`\n"
        "- TensorBoard logs: `training_logs/`\n"
        "- Evaluation results: `evaluation_results.json`\n",
        encoding='utf-8'
    )
    return paths


def _create_training_environment(config, run_away_after_maul=True):
    global _PARALLEL_MP_INITIALIZED
    try:
        env_name = ENVIRONMENT_IDS[config.environment]
    except KeyError as error:
        raise ValueError(
            f"Unknown environment '{config.environment}'. "
            f"Choose one of {sorted(ENVIRONMENT_IDS)}."
        ) from error
    environment_kwargs = {}
    if config.environment == 'hunt':
        environment_kwargs['run_away_after_maul'] = run_away_after_maul

    if config.num_parallel_envs <= 1:
        py_env = TFStagHuntWrapper(
            env_name=env_name,
            seed=config.seed,
            **environment_kwargs,
        )
        return py_env, tf_py_environment.TFPyEnvironment(py_env), False

    if not _PARALLEL_MP_INITIALIZED:
        tf_multiprocessing.enable_interactive_mode()
        _PARALLEL_MP_INITIALIZED = True
    seeds = [config.seed + environment_index for environment_index in range(config.num_parallel_envs)]
    env_constructors = [
        lambda seed=seed: ParallelStagHuntEnv(
            env_name=env_name,
            seed=seed,
            **environment_kwargs,
        )
        for seed in seeds
    ]
    py_env = parallel_py_environment.ParallelPyEnvironment(
        env_constructors,
        start_serially=True,
        blocking=False,
        flatten=True,
    )
    return py_env, tf_py_environment.TFPyEnvironment(py_env), True


def _activate_curriculum(config, py_env, tf_env, memory, agent, is_parallel):
    if config.environment != 'hunt':
        return py_env, tf_env, memory, None
    if is_parallel:
        py_env.close()
        py_env, tf_env, _ = _create_training_environment(
            config,
            run_away_after_maul=False,
        )
        memory = MemoryManager(
            tf_env,
            agent,
            max_length=config.rollout_steps + 1,
        )
        return py_env, tf_env, memory, tf_env.reset()

    py_env.set_stag_run_away_after_maul(False)
    return py_env, tf_env, memory, None


def _build_agent(config, observation_spec, action_spec):
    def build_actor():
        return ActorNetwork(observation_spec, action_spec, hidden_sizes=(128, 128))

    if config.critic_observation not in {'local', 'joint'}:
        raise ValueError("critic_observation must be 'local' or 'joint'.")

    critic_spec = observation_spec
    critic_output_size = 1
    if config.critic_observation == 'joint':
        critic_spec = tensor_spec.TensorSpec(
            shape=(observation_spec.shape[-1] * 2,),
            dtype=observation_spec.dtype,
            name='joint_observation',
        )
        critic_output_size = 2

    def build_critic():
        return CriticNetwork(
            critic_spec,
            hidden_sizes=(128, 128),
            output_size=critic_output_size,
        )

    architecture = config.agent_architecture
    if architecture == 'shared_shared':
        actors = [build_actor()]
        critics = [build_critic()]
    elif architecture == 'separate_actors_shared_critic':
        actors = [build_actor(), build_actor()]
        shared_critic = build_critic()
        critics = [shared_critic, shared_critic]
    elif architecture == 'separate_actors_separate_critics':
        actors = [build_actor(), build_actor()]
        critics = [build_critic(), build_critic()]
    else:
        raise ValueError(
            f'Unknown agent architecture: {architecture}. '
            'Use shared_shared, separate_actors_shared_critic, or '
            'separate_actors_separate_critics.'
        )

    trainers = [
        CustomPPO(
            actor_net=actor,
            critic_net=critic,
            lr=config.learning_rate,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_start,
            value_normalization=config.value_normalization,
            value_clipping=config.value_clipping,
            value_clip_epsilon=config.value_clip_epsilon,
        )
        for actor, critic in zip(actors, critics)
    ]
    if config.value_normalization and architecture in {
        'shared_shared', 'separate_actors_shared_critic'
    }:
        shared_normalizer = trainers[0].value_normalizer
        for trainer in trainers[1:]:
            trainer.value_normalizer = shared_normalizer
    return MultiAgentPPO(trainers, architecture)


def train_agent(experiment_name=None, config=None, output_dir=None):
    config = config or TrainingConfig()
    tf.keras.utils.set_random_seed(config.seed)
    experiment_paths = _create_experiment_paths(config, experiment_name, root_dir=output_dir)
    print(f"Experiment outputs will be saved to: {experiment_paths['root']}")
    print(f"Random seed: {config.seed}")
    
    print("1. Initializing Environment...")
    py_env, tf_env, is_parallel = _create_training_environment(config)

    print("2. Building Neural Networks...")
    if is_parallel:
        network_observation_spec = tensor_spec.TensorSpec(
            shape=(tf_env.observation_spec().shape[-1],),
            dtype=tf.float32,
            name='observation',
        )
        network_action_spec = tensor_spec.BoundedTensorSpec(
            shape=(), dtype=tf.int32, minimum=0, maximum=4, name='action'
        )
    else:
        network_observation_spec = tf_env.observation_spec()
        network_action_spec = tf_env.action_spec()
    agent = _build_agent(
        config,
        network_observation_spec,
        network_action_spec,
    )

    print("3. Compiling the CUSTOM PPO Agent...")
    train_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")    
    heatmap_globale_a = tf.Variable(tf.zeros((config.map_size, config.map_size), dtype=tf.int32), trainable=False, name="heatmap_globale_a")
    heatmap_globale_b = tf.Variable(tf.zeros((config.map_size, config.map_size), dtype=tf.int32), trainable=False, name="heatmap_globale_b")

    entropy_scheduler = EntropyScheduler(
        agent=agent, 
        start_value=config.entropy_start, 
        end_value=config.entropy_end, 
        decay_steps=config.entropy_decay_steps 
    )

    train_summary_writer = tf.summary.create_file_writer(str(experiment_paths['training_logs']))
    memory = MemoryManager(tf_env, agent, max_length=config.rollout_steps + 1)
    
    checkpoint_items = {
        'global_step': train_step_counter,
        'heatmap_globale_a': heatmap_globale_a,
        'heatmap_globale_b': heatmap_globale_b,
    }
    if agent.architecture == 'shared_shared':
        checkpoint_items.update({
            'actor_network': agent.actor,
            'critic_network': agent.critic,
            'optimizer': agent.optimizer,
        })
    else:
        for index, trainer in enumerate(agent.trainers):
            checkpoint_items[f'actor_{index}'] = trainer.actor
            checkpoint_items[f'critic_{index}'] = trainer.critic
            checkpoint_items[f'optimizer_{index}'] = trainer.optimizer
    if config.value_normalization:
        saved_normalizers = set()
        for index, trainer in enumerate(agent.trainers):
            normalizer = trainer.value_normalizer
            if normalizer is None or id(normalizer) in saved_normalizers:
                continue
            saved_normalizers.add(id(normalizer))
            checkpoint_items[f'value_norm_mean_{index}'] = normalizer.mean
            checkpoint_items[f'value_norm_variance_{index}'] = normalizer.variance
            checkpoint_items[f'value_norm_count_{index}'] = normalizer.count
    train_checkpointer = common.Checkpointer(
        ckpt_dir=str(experiment_paths['checkpoints']),
        max_to_keep=3,
        **checkpoint_items,
    )

    train_checkpointer.initialize_or_restore()
    start_step = train_step_counter.numpy()
    print(f"--> Restored state. Global starting step: {start_step}")
    print("=== BEGINNING TRAINING LOOP ===")

    hard_mode_activated = False
    time_step = tf_env.reset()
    training_metrics = []
    
    for i in range(config.num_iterations):
        current_global_step = train_step_counter.numpy()
        print(f"\n--- Iteration {i+1} (Global Step: {current_global_step}) ---")
        current_learning_rate = _annealed_learning_rate(config, i)
        agent.set_learning_rate(current_learning_rate)
        
        # --- Hyperparameter and curriculum update ---
        current_entropy = entropy_scheduler.step(current_global_step)

        if (
            config.environment == 'hunt'
            and current_global_step >= config.curriculum_threshold
            and not hard_mode_activated
        ):
            print(">>> CURRICULUM LEARNING: Difficulty increased! Unforgiving stag enabled.")
            py_env, tf_env, memory, curriculum_time_step = _activate_curriculum(
                config,
                py_env,
                tf_env,
                memory,
                agent,
                is_parallel,
            )
            if curriculum_time_step is not None:
                time_step = curriculum_time_step
            hard_mode_activated = True

        # --- A. Data collection ---
        print("A) Data collection (multiple rollout: 4 consecutive episodes)...")
        time_step, fast_heatmap_a, fast_heatmap_b, stats = collect_rollout(agent, tf_env, memory, time_step, config)

        # --- B/C. PPO training ---
        print("B/C) Trajectory extraction and Custom PPO training in progress...")
        trajectories = memory.get_sequential_data()
        heatmap_globale_a.assign_add(fast_heatmap_a)
        heatmap_globale_b.assign_add(fast_heatmap_b)
        
        train_loss, custom_return, agent_returns, diagnostics = train_ppo_epochs(
            agent, trajectories, current_global_step, config
        )
        memory.clear_buffer()
        training_metrics.append({
            'return': float(custom_return.numpy()),
            'agent_a_return': float(tf.reduce_mean(agent_returns[:, 0]).numpy()),
            'agent_b_return': float(tf.reduce_mean(agent_returns[:, 1]).numpy()),
            'loss': float(train_loss.numpy() if hasattr(train_loss, 'numpy') else train_loss),
            'entropy': float(current_entropy),
            'stags_caught': float(stats['stags']),
            'plants_eaten': float(stats['plants']),
            'maulings_sustained': float(stats['mauling']),
            'learning_rate': float(diagnostics['learning_rate'].numpy()),
            'approx_kl': float(diagnostics['approx_kl'].numpy()),
            'clip_fraction': float(diagnostics['clip_fraction'].numpy()),
            'policy_loss': float(diagnostics['policy_loss'].numpy()),
            'value_loss': float(diagnostics['value_loss'].numpy()),
            'policy_entropy': float(diagnostics['entropy'].numpy()),
            'gradient_norm': float(diagnostics['gradient_norm'].numpy()),
        })

        # --- D. Logging and saving ---
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss/Total_Loss', train_loss, step=current_global_step)
            tf.summary.scalar('Loss/Entropy_Coefficient', current_entropy, step=current_global_step)
            tf.summary.scalar('Metrics/Custom_Average_Return', custom_return, step=current_global_step)
            tf.summary.scalar(
                'Metrics/Agent_A_Return',
                tf.reduce_mean(agent_returns[:, 0]),
                step=current_global_step,
            )
            tf.summary.scalar(
                'Metrics/Agent_B_Return',
                tf.reduce_mean(agent_returns[:, 1]),
                step=current_global_step,
            )
            tf.summary.scalar('Training/Learning_Rate', current_learning_rate, step=current_global_step)
            tf.summary.scalar('Behavior/Stags_Caught', stats['stags'], step=current_global_step)
            tf.summary.scalar('Behavior/Plants_Eaten', stats['plants'], step=current_global_step)
            tf.summary.scalar('Behavior/Maulings_Sustained', stats['mauling'], step=current_global_step)
            
            if current_global_step % config.log_interval == 0:
                print(f"Step {current_global_step} | Current entropy: {current_entropy:.4f}")
                tf.summary.image('Exploration/Heatmap_Agent_A', generate_annotated_heatmap(heatmap_globale_a.numpy()), step=current_global_step)
                tf.summary.image('Exploration/Heatmap_Agent_B', generate_annotated_heatmap(heatmap_globale_b.numpy()), step=current_global_step)
        
        train_step_counter.assign_add(1)
        step = train_step_counter.numpy()
        print(f"Global Step: {step} | Custom PPO Loss: {train_loss:.4f}")

        if step > 0 and step % config.save_interval == 0:
            train_checkpointer.save(global_step=step)
            gc.collect()

    if is_parallel:
        py_env.close()

    total_stags = sum(iteration['stags_caught'] for iteration in training_metrics)
    total_plants = sum(iteration['plants_eaten'] for iteration in training_metrics)
    total_maulings = sum(iteration['maulings_sustained'] for iteration in training_metrics)
    convergence_iteration_a = _find_convergence_iteration(
        training_metrics, config, reward_key='agent_a_return'
    )
    convergence_iteration_b = _find_convergence_iteration(
        training_metrics, config, reward_key='agent_b_return'
    )
    convergence_iteration = (
        max(convergence_iteration_a, convergence_iteration_b)
        if convergence_iteration_a is not None and convergence_iteration_b is not None
        else None
    )
    (experiment_paths['root'] / 'training_results.json').write_text(
        json.dumps({
            'experiment_id': experiment_paths['experiment_id'],
            'seed': config.seed,
            'iterations': len(training_metrics),
            'total_stags_caught': total_stags,
            'total_plants_eaten': total_plants,
            'total_maulings_sustained': total_maulings,
            'convergence_iteration': convergence_iteration,
            'convergence_iteration_agent_a': convergence_iteration_a,
            'convergence_iteration_agent_b': convergence_iteration_b,
            'convergence_definition': {
                'reward_rate_threshold': config.convergence_reward_rate,
                'stable_window': config.convergence_window,
                'max_reward_rate_std': config.convergence_max_rate_std,
            },
            'heatmap_agent_a': heatmap_globale_a.numpy().tolist(),
            'heatmap_agent_b': heatmap_globale_b.numpy().tolist(),
        }, indent=2),
        encoding='utf-8'
    )

    print("Training Completed successfully!")
    return training_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PPO agent in an isolated experiment directory.')
    parser.add_argument(
        '--experiment-name',
        required=True,
        help='Readable experiment prefix. Hyperparameter hash and timestamp are added automatically.'
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, help='Seed for one training run.')
    seed_group.add_argument(
        '--seeds',
        help='Comma-separated seeds for independent sequential training runs.'
    )
    parser.add_argument('--num-iterations', type=int)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--num-parallel-envs', type=int)
    parser.add_argument('--environment', choices=sorted(ENVIRONMENT_IDS))
    parser.add_argument('--agent-architecture', choices=sorted(MultiAgentPPO.VALID_ARCHITECTURES))
    parser.add_argument('--critic-observation', choices=('local', 'joint'))
    parser.add_argument('--reward-sharing-coef', type=float)
    parser.add_argument('--minibatch-size', type=int)
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--curriculum-threshold', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--map-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--learning-rate-end', type=float)
    parser.add_argument('--clip-epsilon', type=float)
    parser.add_argument('--value-loss-coef', type=float)
    parser.add_argument('--value-normalization', action='store_true', default=None)
    parser.add_argument('--value-clipping', action='store_true', default=None)
    parser.add_argument('--value-clip-epsilon', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--gae-lambda', type=float)
    parser.add_argument('--entropy-start', type=float)
    parser.add_argument('--entropy-end', type=float)
    parser.add_argument('--entropy-decay-steps', type=int)
    parser.add_argument('--convergence-reward-rate', type=float)
    parser.add_argument('--convergence-window', type=int)
    parser.add_argument('--convergence-max-rate-std', type=float)
    args = parser.parse_args()
    config_overrides = {
        field.name: getattr(args, field.name)
        for field in TrainingConfig.__dataclass_fields__.values()
        if getattr(args, field.name) is not None
    }
    base_config = replace(TrainingConfig(), **config_overrides)

    if args.seeds is None:
        seeds = [base_config.seed]
    else:
        try:
            seeds = [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]
        except ValueError as error:
            parser.error(f'--seeds must be a comma-separated list of integers: {error}')
        if not seeds:
            parser.error('--seeds must contain at least one integer.')
        if len(seeds) != len(set(seeds)):
            parser.error('--seeds must not contain duplicate values.')

    if len(seeds) == 1:
        train_agent(
            experiment_name=args.experiment_name,
            config=replace(base_config, seed=seeds[0])
        )
    else:
        group_id = _build_experiment_id(base_config, args.experiment_name)
        group_dir = Path('./experiments') / group_id
        if group_dir.exists():
            raise FileExistsError(
                f'Experiment directory already exists: {group_dir}. '
                'Choose a different --experiment-name.'
            )

        group_dir.mkdir(parents=True)
        (group_dir / 'metadata.json').write_text(json.dumps({
            'experiment_id': group_id,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'type': 'multi_seed_training_group',
            'seeds': seeds,
            'hyperparameters': vars(base_config),
        }, indent=2), encoding='utf-8')
        (group_dir / 'README.md').write_text(
            f'# Training group `{group_id}`\n\n'
            f'This experiment aggregates independent training runs for seeds: {seeds}.\n\n'
            '- Per-seed artifacts: `runs/seed_<seed>/`\n'
            '- Aggregate results: `training_results.json`\n'
            '- Group metadata: `metadata.json`\n',
            encoding='utf-8'
        )

        training_runs = []
        run_totals = []
        convergence_iterations = []
        convergence_iterations_a = []
        convergence_iterations_b = []
        heatmap_sum_a = None
        heatmap_sum_b = None
        for seed in seeds:
            run_config = replace(base_config, seed=seed)
            run_dir = group_dir / 'runs' / f'seed_{seed}'
            print(f'=== Starting training run with seed {seed} ===')
            metrics = train_agent(
                experiment_name=f'{group_id}_seed_{seed}',
                config=run_config,
                output_dir=run_dir
            )
            training_runs.append(metrics)

            run_results = json.loads((run_dir / 'training_results.json').read_text(encoding='utf-8'))
            run_totals.append(run_results)
            if run_results['convergence_iteration'] is not None:
                convergence_iterations.append(run_results['convergence_iteration'])
            if run_results['convergence_iteration_agent_a'] is not None:
                convergence_iterations_a.append(run_results['convergence_iteration_agent_a'])
            if run_results['convergence_iteration_agent_b'] is not None:
                convergence_iterations_b.append(run_results['convergence_iteration_agent_b'])
            run_heatmap_a = np.asarray(run_results['heatmap_agent_a'], dtype=int)
            run_heatmap_b = np.asarray(run_results['heatmap_agent_b'], dtype=int)
            heatmap_sum_a = run_heatmap_a if heatmap_sum_a is None else heatmap_sum_a + run_heatmap_a
            heatmap_sum_b = run_heatmap_b if heatmap_sum_b is None else heatmap_sum_b + run_heatmap_b

            tf.keras.backend.clear_session()
            gc.collect()

        metric_names = [
            'return', 'agent_a_return', 'agent_b_return', 'loss', 'entropy', 'stags_caught',
            'plants_eaten', 'maulings_sustained', 'learning_rate', 'approx_kl',
            'clip_fraction', 'policy_loss', 'value_loss', 'policy_entropy', 'gradient_norm'
        ]
        metric_curves = {
            metric: np.asarray([
                [iteration[metric] for iteration in run]
                for run in training_runs
            ], dtype=float)
            for metric in metric_names
        }
        final_metrics = {
            metric: np.asarray([
                run[-1][metric] if run else float('nan')
                for run in training_runs
            ], dtype=float)
            for metric in metric_names
        }
        aggregate_metrics = {}
        for metric in metric_names:
            curve = metric_curves[metric]
            final_values = final_metrics[metric]
            aggregate_metrics[metric] = {
                'final_mean': float(np.nanmean(final_values)),
                'final_std': float(np.nanstd(final_values, ddof=1)) if len(final_values) > 1 else 0.0,
                'mean_by_iteration': np.nanmean(curve, axis=0).tolist(),
                'std_by_iteration': (
                    np.nanstd(curve, axis=0, ddof=1) if len(training_runs) > 1
                    else np.zeros(curve.shape[1])
                ).tolist(),
                'final_by_seed': {
                    str(seed): float(value)
                    for seed, value in zip(seeds, final_values)
                },
            }

        aggregate_results = {
            'experiment_id': group_id,
            'seeds': seeds,
            'iterations': len(training_runs[0]) if training_runs else 0,
            'metrics': aggregate_metrics,
            'total_stags_caught': sum(run['total_stags_caught'] for run in run_totals),
            'total_plants_eaten': sum(run['total_plants_eaten'] for run in run_totals),
            'total_maulings_sustained': sum(run['total_maulings_sustained'] for run in run_totals),
            'convergence_iteration_mean': (
                float(np.mean(convergence_iterations)) if convergence_iterations else None
            ),
            'convergence_iteration_std': (
                float(np.std(convergence_iterations, ddof=1))
                if len(convergence_iterations) > 1 else 0.0
            ),
            'converged_seeds': len(convergence_iterations),
            'convergence_iteration_agent_a_mean': (
                float(np.mean(convergence_iterations_a)) if convergence_iterations_a else None
            ),
            'convergence_iteration_agent_b_mean': (
                float(np.mean(convergence_iterations_b)) if convergence_iterations_b else None
            ),
            'converged_seeds_agent_a': len(convergence_iterations_a),
            'converged_seeds_agent_b': len(convergence_iterations_b),
            'convergence_definition': {
                'reward_rate_threshold': base_config.convergence_reward_rate,
                'stable_window': base_config.convergence_window,
                'max_reward_rate_std': base_config.convergence_max_rate_std,
            },
            'heatmap_agent_a': heatmap_sum_a.tolist() if heatmap_sum_a is not None else [],
            'heatmap_agent_b': heatmap_sum_b.tolist() if heatmap_sum_b is not None else [],
        }
        (group_dir / 'training_results.json').write_text(
            json.dumps(aggregate_results, indent=2), encoding='utf-8'
        )
        print('Aggregate final metrics:')
        for metric, values in aggregate_metrics.items():
            print(
                f'  {metric}: mean={values["final_mean"]:.4f}, '
                f'std={values["final_std"]:.4f}'
            )