"""Training orchestration: one seed (`train_agent`) and a multi-seed group (`train_seed_group`)."""
import gc
import json
from dataclasses import replace
from pathlib import Path

import tensorflow as tf
from tf_agents.utils import common

from training.agent import (
    EntropyScheduler,
    annealed_learning_rate,
    build_agent,
    checkpoint_items,
    network_specs,
)
from training.config import TrainingConfig
from training.environment import (
    activate_curriculum,
    create_training_environment,
    set_stag_follow_probability,
    stag_follow_probability,
)
from training.experiment import build_experiment_id, create_experiment_paths, write_group_metadata
from training.monitoring import (
    TensorBoardLogger,
    custom_return_window_is_stable,
    group_results,
    iteration_metrics,
    run_results,
)
from training.ppo import train_ppo_epochs
from training.rollout import MemoryManager, collect_rollout


def _uses_hunt_curriculum(config):
    return config.curriculum_enabled and config.environment == 'hunt'


def train_agent(experiment_name=None, config=None, output_dir=None, resume=False,
                experiments_root='./experiments'):
    """Runs PPO training for one seed; writes checkpoints, logs and training_results.json."""
    config = config or TrainingConfig()
    tf.keras.utils.set_random_seed(config.seed)
    paths = create_experiment_paths(
        config, experiment_name, root_dir=output_dir, resume=resume, experiments_root=experiments_root
    )
    print(f"Experiment outputs will be saved to: {paths['root']}")
    print(f"Random seed: {config.seed}")

    print("1. Initializing Environment...")
    py_env, tf_env, is_parallel = create_training_environment(config)

    print("2. Building Neural Networks...")
    observation_spec, action_spec = network_specs(tf_env, is_parallel)
    agent = build_agent(config, observation_spec, action_spec)

    print("3. Compiling the CUSTOM PPO Agent...")
    train_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
    heatmap_shape = (config.map_size, config.map_size)
    global_heatmap_a = tf.Variable(tf.zeros(heatmap_shape, dtype=tf.int32), trainable=False, name="heatmap_globale_a")
    global_heatmap_b = tf.Variable(tf.zeros(heatmap_shape, dtype=tf.int32), trainable=False, name="heatmap_globale_b")
    entropy_scheduler = EntropyScheduler(
        agent, start_value=config.entropy_start, end_value=config.entropy_end, decay_steps=config.entropy_decay_steps
    )
    logger = TensorBoardLogger(paths['training_logs'], config.log_interval)
    memory = MemoryManager(tf_env, agent, max_length=config.rollout_steps + 1)
    checkpointer = common.Checkpointer(
        ckpt_dir=str(paths['checkpoints']),
        max_to_keep=1000,
        **checkpoint_items(agent, config, train_step_counter, global_heatmap_a, global_heatmap_b),
    )
    checkpointer.initialize_or_restore()
    start_step = train_step_counter.numpy()
    print(f"--> Restored state. Global starting step: {start_step}")
    print("=== BEGINNING TRAINING LOOP ===")

    hard_mode_activated = False
    time_step = tf_env.reset()
    training_metrics = []

    for iteration in range(start_step, config.num_iterations):
        step = train_step_counter.numpy()
        print(f"\n--- Iteration {iteration + 1} (Global Step: {step}) ---")
        learning_rate = annealed_learning_rate(config, iteration)
        agent.set_learning_rate(learning_rate)
        entropy_coefficient = entropy_scheduler.step(step)

        if _uses_hunt_curriculum(config) and step >= config.curriculum_threshold and not hard_mode_activated:
            print(">>> CURRICULUM LEARNING: Difficulty increased! Unforgiving stag enabled.")
            py_env, tf_env, memory, reset_step = activate_curriculum(config, py_env, tf_env, memory, agent, is_parallel)
            if reset_step is not None:
                time_step = reset_step
            hard_mode_activated = True

        follow_probability = stag_follow_probability(config, step)
        if config.stag_follow_curriculum_enabled and config.environment == 'hunt':
            set_stag_follow_probability(py_env, is_parallel, follow_probability)

        print("A) Data collection")
        time_step, rollout_heatmap_a, rollout_heatmap_b, stats = collect_rollout(agent, tf_env, memory, time_step, config)

        print("B/C) Trajectory extraction and Custom PPO training in progress...")
        trajectories = memory.get_sequential_data()
        global_heatmap_a.assign_add(rollout_heatmap_a)
        global_heatmap_b.assign_add(rollout_heatmap_b)
        train_loss, custom_return, agent_returns, diagnostics = train_ppo_epochs(agent, trajectories, step, config)
        memory.clear_buffer()

        metric = iteration_metrics(
            custom_return, agent_returns, train_loss, entropy_coefficient,
            config.stag_follows if follow_probability is None else follow_probability,
            stats, diagnostics,
        )
        training_metrics.append(metric)
        stable = custom_return_window_is_stable(training_metrics, config)

        logger.log_iteration(
            step, metric, stats, diagnostics, agent_returns, rollout_heatmap_a, rollout_heatmap_b,
            custom_return, learning_rate, entropy_coefficient, stable,
        )
        logger.log_periodic(
            step, agent, diagnostics, global_heatmap_a.numpy(), global_heatmap_b.numpy(),
            rollout_heatmap_a, rollout_heatmap_b,
        )
        if step % config.log_interval == 0:
            print(f"Step {step} | Current entropy: {entropy_coefficient:.4f}")

        train_step_counter.assign_add(1)
        step = train_step_counter.numpy()
        print(f"Global Step: {step} | Custom PPO Loss: {train_loss:.4f}")
        if step > 0 and step % config.save_interval == 0:
            checkpointer.save(global_step=step)
            logger.flush()
            gc.collect()

    if config.stag_follow_curriculum_enabled and config.environment == 'hunt':
        set_stag_follow_probability(py_env, is_parallel, None)
    if is_parallel:
        py_env.close()

    (paths['root'] / 'training_results.json').write_text(
        json.dumps(
            run_results(paths['experiment_id'], config, training_metrics,
                        global_heatmap_a.numpy(), global_heatmap_b.numpy()),
            indent=2,
        ),
        encoding='utf-8',
    )
    print("Training Completed successfully!")
    return training_metrics


def train_seed_group(experiment_name, base_config, seeds, experiments_root):
    """Trains one run per seed under <group>/runs/seed_<seed>/ and aggregates their results."""
    group_id = build_experiment_id(base_config, experiment_name)
    group_dir = Path(experiments_root) / group_id
    if group_dir.exists():
        raise FileExistsError(
            f'Experiment directory already exists: {group_dir}. Choose a different --experiment-name.'
        )
    group_dir.mkdir(parents=True)
    write_group_metadata(group_dir, group_id, base_config, seeds)

    training_runs, run_totals = [], []
    for seed in seeds:
        run_dir = group_dir / 'runs' / f'seed_{seed}'
        print(f'=== Starting training run with seed {seed} ===')
        training_runs.append(train_agent(
            experiment_name=f'{group_id}_seed_{seed}',
            config=replace(base_config, seed=seed),
            output_dir=run_dir,
        ))
        run_totals.append(json.loads((run_dir / 'training_results.json').read_text(encoding='utf-8')))
        tf.keras.backend.clear_session()
        gc.collect()

    aggregate = group_results(group_id, base_config, seeds, training_runs, run_totals)
    (group_dir / 'training_results.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print('Aggregate final metrics:')
    for metric, values in aggregate['metrics'].items():
        print(f'  {metric}: mean={values["final_mean"]:.4f}, std={values["final_std"]:.4f}')
    return aggregate
