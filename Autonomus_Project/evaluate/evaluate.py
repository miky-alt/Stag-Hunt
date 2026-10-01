import os
import argparse
import json
from pathlib import Path
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
import numpy as np
import tensorflow as tf

from tf_agents.environments import wrappers, tf_py_environment
from tf_agents.utils import common

# Import our modules.
from envs.tf_wrapper import TFStagHuntWrapper
from agents.networks import ActorNetwork

# =========================================================
# Wrapper (must remain identical to train.py).
# =========================================================
class TitaniumWrapper(wrappers.PyEnvironmentBaseWrapper):
    def _step(self, action):
        return self._sanitize(self._env.step(action))
    def _reset(self):
        return self._sanitize(self._env.reset())
    def _sanitize(self, ts):
        obs = np.asarray(ts.observation, dtype=np.float32)
        rew = np.asarray(ts.reward, dtype=np.float32).reshape(-1)
        disc = np.asarray(ts.discount, dtype=np.float32)
        return ts._replace(observation=obs, reward=rew, discount=disc)


DEFAULT_EVAL_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
ENVIRONMENT_IDS = {
    'hunt': 'StagHunt-Hunt-v0',
    'harvest': 'StagHunt-Harvest-v0',
    'escalation': 'StagHunt-Escalation-v0',
}


def _summarize_episode_rewards(agent_rewards):
    """Summarize complete-episode rewards separately for both agents."""
    rewards = np.asarray(agent_rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[1] != 2 or rewards.shape[0] == 0:
        raise ValueError('Expected a non-empty [episode, agent] reward array.')
    team_rewards = np.sum(rewards, axis=1)
    return {
        'mean_reward': float(np.mean(team_rewards)),
        'std_reward': float(np.std(team_rewards, ddof=1)) if len(team_rewards) > 1 else 0.0,
        'mean_agent_a_reward': float(np.mean(rewards[:, 0])),
        'std_agent_a_reward': float(np.std(rewards[:, 0], ddof=1)) if len(rewards) > 1 else 0.0,
        'mean_agent_b_reward': float(np.mean(rewards[:, 1])),
        'std_agent_b_reward': float(np.std(rewards[:, 1], ddof=1)) if len(rewards) > 1 else 0.0,
    }


def _sample_uniform_actions(time_step, action_spec):
    """Sample one independent uniform discrete action per agent."""
    batch_size = int(np.asarray(time_step.observation.numpy()).shape[0])
    return tf.random.uniform(
        shape=(batch_size,),
        minval=int(action_spec.minimum),
        maxval=int(action_spec.maximum) + 1,
        dtype=tf.int32,
    )


def evaluate_agent(
    experiment_name,
    episodes_per_seed=10,
    seeds=None,
    render=False,
    policy='checkpoint',
):
    if policy not in {'checkpoint', 'random_uniform'}:
        raise ValueError("policy must be 'checkpoint' or 'random_uniform'.")
    if episodes_per_seed < 1:
        raise ValueError('episodes_per_seed must be at least 1.')
    seeds = list(DEFAULT_EVAL_SEEDS if seeds is None else seeds)
    if not seeds:
        raise ValueError('At least one evaluation seed is required.')

    experiment_dir = Path('./experiments') / experiment_name
    metadata_path = experiment_dir / 'metadata.json'
    checkpoint_dir = experiment_dir / 'checkpoints'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    if policy == 'checkpoint' and not checkpoint_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoints not found: {checkpoint_dir}")

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    hyperparameters = metadata['hyperparameters']
    environment = hyperparameters.get('environment', 'hunt')
    try:
        env_name = ENVIRONMENT_IDS[environment]
    except KeyError as error:
        raise ValueError(f"Unknown environment in metadata: {environment}") from error
    print(f"Experiment: {metadata['experiment_id']}")
    print(f"Evaluation policy: {policy}")
    print(f"Hyperparameters: {json.dumps(hyperparameters, sort_keys=True)}")

    print(f"Evaluation seeds: {seeds}")
    print(f"Episodes per seed: {episodes_per_seed}")

    print("1. Initializing evaluation environment...")
    # Build the network from the same specs used by the training environment.
    base_py_env = TFStagHuntWrapper(
        env_name=env_name,
        render_mode="human" if render else None,
        **({'run_away_after_maul': False} if environment == 'hunt' else {})
    )
    wrapped_py_env = TitaniumWrapper(base_py_env)
    tf_env = tf_py_environment.TFPyEnvironment(wrapped_py_env)

    actor_net = None
    train_step_counter = tf.Variable(0, dtype=tf.int64)
    if policy == 'checkpoint':
        print("2. Rebuilding the actor network only...")
        actor_net = ActorNetwork(
            input_tensor_spec=tf_env.observation_spec(),
            output_tensor_spec=tf_env.action_spec(),
            hidden_sizes=(128, 128)
        )
    else:
        print("2. Using independent uniform random actions...")
    wrapped_py_env.close()

    if policy == 'checkpoint':
        print("3. Loading checkpoint state...")
        checkpointer = common.Checkpointer(
            ckpt_dir=str(checkpoint_dir),
            actor_network=actor_net,
            global_step=train_step_counter
        )
        status = checkpointer.initialize_or_restore()
        status.expect_partial()
        print(f"--> Model restored successfully from step: {train_step_counter.numpy()}")

    map_size = int(metadata['hyperparameters'].get('map_size', 5))
    heatmap_a = np.zeros((map_size, map_size), dtype=int)
    heatmap_b = np.zeros((map_size, map_size), dtype=int)
    behavior_totals = {'stags_caught': 0, 'plants_eaten': 0, 'maulings_sustained': 0}

    print("4. Starting evaluation...")
    evaluation_results = []
    seed_summaries = []
    all_team_rewards = []
    all_agent_rewards = [[], []]

    for seed in seeds:
        tf.keras.utils.set_random_seed(seed)
        base_py_env = TFStagHuntWrapper(
            env_name=env_name,
            render_mode="human" if render else None,
            seed=seed,
            **({'run_away_after_maul': False} if environment == 'hunt' else {})
        )
        wrapped_py_env = TitaniumWrapper(base_py_env)
        tf_env = tf_py_environment.TFPyEnvironment(wrapped_py_env)
        seed_team_rewards = []
        seed_agent_rewards = [[], []]

        for episode in range(episodes_per_seed):
            print(f"Seed {seed} | Episode {episode + 1}/{episodes_per_seed}")
            time_step = tf_env.reset()
            episode_agent_rewards = np.zeros(2, dtype=np.float32)
            step_count = 0
            max_steps = 100

            while not time_step.is_last()[0] and step_count < max_steps:
                step_count += 1
                if policy == 'checkpoint':
                    action_distribution, _ = actor_net(time_step.observation)
                    action = tf.cast(action_distribution.sample(), tf.int32)
                else:
                    action = _sample_uniform_actions(time_step, tf_env.action_spec())
                time_step = tf_env.step(action)
                step_rewards = np.asarray(time_step.reward.numpy(), dtype=np.float32).reshape(-1)
                if step_rewards.size != 2:
                    raise ValueError(
                        f'Expected one reward per agent, received shape {step_rewards.shape}.'
                    )
                episode_agent_rewards += step_rewards
                episode_reward = float(np.sum(episode_agent_rewards))
                step_team_reward = float(np.sum(step_rewards))

                # Agent A's row also carries the partner's (agent B) position.
                own_and_partner_position = time_step.observation.numpy()[0]
                xa = min(int(own_and_partner_position[0]), map_size - 1)
                ya = min(int(own_and_partner_position[1]), map_size - 1)
                xb = min(int(own_and_partner_position[2]), map_size - 1)
                yb = min(int(own_and_partner_position[3]), map_size - 1)
                heatmap_a[xa, ya] += 1
                heatmap_b[xb, yb] += 1

                if step_team_reward >= 10.0:
                    behavior_totals['stags_caught'] += 1
                elif 1.0 <= step_team_reward <= 2.0:
                    behavior_totals['plants_eaten'] += 1
                elif step_team_reward < 0.0:
                    behavior_totals['maulings_sustained'] += 1

                if render:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    base_py_env.env.render()
                    print(
                        f"Step: {step_count} | Selected actions: {action.numpy()} "
                        f"| Immediate rewards: {step_rewards.tolist()}"
                    )
                    time.sleep(0.3)

            if render:
                os.system('cls' if os.name == 'nt' else 'clear')
                base_py_env.env.render()
            print(
                f"Seed {seed} | Episode {episode + 1} finished. "
                f"Total reward: {episode_reward:.2f}, Steps: {step_count}"
            )
            episode_agent_rewards = episode_agent_rewards.astype(float)
            episode_reward = float(np.sum(episode_agent_rewards))
            seed_team_rewards.append(episode_reward)
            seed_agent_rewards[0].append(episode_agent_rewards[0])
            seed_agent_rewards[1].append(episode_agent_rewards[1])
            all_team_rewards.append(episode_reward)
            all_agent_rewards[0].append(episode_agent_rewards[0])
            all_agent_rewards[1].append(episode_agent_rewards[1])
            evaluation_results.append({
                'seed': seed,
                'episode': episode + 1,
                'reward': episode_reward,
                'agent_a_reward': float(episode_agent_rewards[0]),
                'agent_b_reward': float(episode_agent_rewards[1]),
                'steps': step_count,
            })

        seed_summaries.append({
            'seed': seed,
            'episodes': len(seed_team_rewards),
            **_summarize_episode_rewards(np.asarray(seed_agent_rewards).T),
        })

        wrapped_py_env.close()

    all_statistics = _summarize_episode_rewards(np.asarray(all_agent_rewards).T)
    mean_reward = all_statistics['mean_reward']
    std_reward = all_statistics['std_reward']
    mean_agent_a_reward = all_statistics['mean_agent_a_reward']
    std_agent_a_reward = all_statistics['std_agent_a_reward']
    mean_agent_b_reward = all_statistics['mean_agent_b_reward']
    std_agent_b_reward = all_statistics['std_agent_b_reward']
    seed_means = [summary['mean_reward'] for summary in seed_summaries]

    (experiment_dir / 'evaluation_results.json').write_text(
        json.dumps({
            'experiment_id': metadata['experiment_id'],
            'checkpoint_step': int(train_step_counter.numpy()),
            'seeds': seeds,
            'episodes_per_seed': episodes_per_seed,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_agent_a_reward': mean_agent_a_reward,
            'std_agent_a_reward': std_agent_a_reward,
            'mean_agent_b_reward': mean_agent_b_reward,
            'std_agent_b_reward': std_agent_b_reward,
            'mean_reward_across_seeds': float(np.mean(seed_means)),
            'std_reward_across_seeds': float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            'by_seed': seed_summaries,
            'episodes': evaluation_results,
            'behavior': {
                'map_size': map_size,
                'total_stags_caught': behavior_totals['stags_caught'],
                'total_plants_eaten': behavior_totals['plants_eaten'],
                'total_maulings_sustained': behavior_totals['maulings_sustained'],
                'heatmap_agent_a': heatmap_a.tolist(),
                'heatmap_agent_b': heatmap_b.tolist(),
            },
        }, indent=2),
        encoding='utf-8'
    )
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    print(f"Agent A mean/std: {mean_agent_a_reward:.4f} / {std_agent_a_reward:.4f}")
    print(f"Agent B mean/std: {mean_agent_b_reward:.4f} / {std_agent_b_reward:.4f}")
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate one isolated training experiment.')
    parser.add_argument('--experiment', required=True, help='Experiment directory name under ./experiments.')
    parser.add_argument(
        '--seeds',
        default=','.join(str(seed) for seed in DEFAULT_EVAL_SEEDS),
        help='Comma-separated evaluation seeds.'
    )
    parser.add_argument(
        '--episodes-per-seed',
        '--episodes',
        dest='episodes_per_seed',
        type=int,
        default=10,
        help='Number of evaluation episodes for each seed.'
    )
    parser.add_argument('--render', action='store_true', help='Render the evaluation episodes.')
    parser.add_argument(
        '--policy',
        choices=('checkpoint', 'random_uniform'),
        default='checkpoint',
        help='Evaluation policy: trained checkpoint or independent uniform random actions.',
    )
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]
    evaluate_agent(
        args.experiment,
        episodes_per_seed=args.episodes_per_seed,
        seeds=seeds,
        render=args.render,
        policy=args.policy,
    )