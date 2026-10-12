"""Evaluates a trained experiment (or the uniform-random floor) and writes evaluation_results.json."""
import json
import os
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from envs.registry import resolve_environment_id
from envs.tf_wrapper import TFStagHuntWrapper
from evaluate.events import behavior_event_names, extract_behavior_event_counts, summarize_behavior_events
from evaluate.policies import actor_actions, build_actor_networks, restore_actors, sample_uniform_actions
from evaluate.statistics import summarize_episode_rewards

DEFAULT_EVAL_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
EPISODE_MAX_STEPS = 100
POLICIES = ('checkpoint', 'random_uniform')


def _environment_kwargs(environment, hyperparameters, grid_size):
    """Wrapper options that reproduce the training game (plus an optional grid override)."""
    kwargs = {
        # Older Hunt runs always used the handcrafted radar and have no field in metadata.
        'handcrafted_features': bool(hyperparameters.get('handcrafted_features', environment == 'hunt')),
    }
    if environment == 'hunt':
        kwargs['stag_follows'] = bool(hyperparameters.get('stag_follows', True))
        kwargs['run_away_after_maul'] = bool(hyperparameters.get('run_away_after_maul', False))
    if grid_size is not None:
        # Out-of-distribution test: coordinates are fed unnormalized.
        kwargs['grid_size'] = (grid_size, grid_size)
    return kwargs


def _validate(policy, episodes_per_seed, render_delay, checkpoint_step, grid_size, seeds):
    if policy not in POLICIES:
        raise ValueError("policy must be 'checkpoint' or 'random_uniform'.")
    if episodes_per_seed < 1:
        raise ValueError('episodes_per_seed must be at least 1.')
    if render_delay < 0:
        raise ValueError('render_delay must be non-negative.')
    if checkpoint_step is not None and checkpoint_step < 0:
        raise ValueError('checkpoint_step must be non-negative.')
    if grid_size is not None and grid_size < 3:
        raise ValueError('grid_size must be at least 3.')
    if not seeds:
        raise ValueError('At least one evaluation seed is required.')


def _render(base_py_env, step_count=None, action=None, step_rewards=None, delay=0.0):
    os.system('cls' if os.name == 'nt' else 'clear')
    base_py_env.env.render()
    if step_count is not None:
        print(f"Step: {step_count} | Selected actions: {action.numpy()} | Immediate rewards: {step_rewards.tolist()}")
        time.sleep(delay)


def _run_episode(tf_env, base_py_env, environment, act, map_size, heatmaps, behavior_totals,
                 render, render_delay):
    """Plays one episode; returns (agent_rewards[2], behavior_events, step_count)."""
    time_step = tf_env.reset()
    episode_agent_rewards = np.zeros(2, dtype=np.float32)
    episode_events = {name: 0.0 for name in behavior_totals}
    step_count = 0
    while not time_step.is_last()[0] and step_count < EPISODE_MAX_STEPS:
        step_count += 1
        action = act(time_step)
        time_step = tf_env.step(action)
        for event_name, count in extract_behavior_event_counts(environment, base_py_env.last_info).items():
            episode_events[event_name] += count
            behavior_totals[event_name] += count
        step_rewards = np.asarray(time_step.reward.numpy(), dtype=np.float32).reshape(-1)
        if step_rewards.size != 2:
            raise ValueError(f'Expected one reward per agent, received shape {step_rewards.shape}.')
        episode_agent_rewards += step_rewards

        # Agent A's observation row also carries the partner's position.
        positions = time_step.observation.numpy()[0]
        heatmaps[0][min(int(positions[0]), map_size - 1), min(int(positions[1]), map_size - 1)] += 1
        heatmaps[1][min(int(positions[2]), map_size - 1), min(int(positions[3]), map_size - 1)] += 1
        if render:
            _render(base_py_env, step_count, action, step_rewards, render_delay)
    if render:
        _render(base_py_env)
    return episode_agent_rewards.astype(float), episode_events, step_count


def evaluate_agent(
    experiment_name,
    episodes_per_seed=10,
    seeds=None,
    render=False,
    policy='checkpoint',
    deterministic=False,
    render_delay=0.3,
    output_filename='evaluation_results.json',
    checkpoint_step=None,
    experiments_root='./experiments',
    grid_size=None,
):
    """Evaluates a saved policy (or uniform random actions) and writes per-episode metrics."""
    seeds = list(DEFAULT_EVAL_SEEDS if seeds is None else seeds)
    _validate(policy, episodes_per_seed, render_delay, checkpoint_step, grid_size, seeds)

    experiment_dir = Path(experiments_root) / experiment_name
    metadata_path = experiment_dir / 'metadata.json'
    checkpoint_dir = experiment_dir / 'checkpoints'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    if policy == 'checkpoint' and not checkpoint_dir.exists():
        raise FileNotFoundError(f"Experiment checkpoints not found: {checkpoint_dir}")

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    hyperparameters = metadata['hyperparameters']
    environment = hyperparameters.get('environment', 'hunt')
    env_name = resolve_environment_id(environment)
    environment_kwargs = _environment_kwargs(environment, hyperparameters, grid_size)
    print(f"Experiment: {metadata['experiment_id']}")
    print(f"Evaluation policy: {policy}")
    print(f"Hyperparameters: {json.dumps(hyperparameters, sort_keys=True)}")
    print(f"Evaluation seeds: {seeds}")
    print(f"Episodes per seed: {episodes_per_seed}")
    if grid_size is not None:
        print(f"Grid size override: {grid_size}x{grid_size}")

    # Build the actors against the specs of the training-equivalent environment.
    print("1. Initializing evaluation environment...")
    spec_env = tf_py_environment.TFPyEnvironment(
        TFStagHuntWrapper(env_name=env_name, render_mode="human" if render else None, **environment_kwargs)
    )
    actor_nets = []
    train_step_counter = tf.Variable(0, dtype=tf.int64)
    if policy == 'checkpoint':
        print("2. Rebuilding the actor network only...")
        actor_nets = build_actor_networks(hyperparameters, spec_env.observation_spec(), spec_env.action_spec())
    else:
        print("2. Using independent uniform random actions...")
    spec_env.close()
    if policy == 'checkpoint':
        print("3. Loading checkpoint state...")
        restore_actors(actor_nets, checkpoint_dir, train_step_counter, checkpoint_step)
        print(f"--> Model restored successfully from step: {train_step_counter.numpy()}")

    map_size = grid_size if grid_size is not None else int(hyperparameters.get('map_size', 5))
    heatmaps = (np.zeros((map_size, map_size), dtype=int), np.zeros((map_size, map_size), dtype=int))
    behavior_totals = {name: 0.0 for name in behavior_event_names(environment)}
    episodes, seed_summaries, all_agent_rewards = [], [], []

    print("4. Starting evaluation...")
    for seed in seeds:
        tf.keras.utils.set_random_seed(seed)
        base_py_env = TFStagHuntWrapper(
            env_name=env_name, render_mode="human" if render else None, seed=seed, **environment_kwargs
        )
        tf_env = tf_py_environment.TFPyEnvironment(base_py_env)
        if policy == 'checkpoint':
            act = lambda ts: actor_actions(actor_nets, ts.observation, deterministic)  # noqa: E731
        else:
            act = lambda ts: sample_uniform_actions(ts, tf_env.action_spec())  # noqa: E731

        seed_agent_rewards, seed_events = [], {name: [] for name in behavior_totals}
        for episode in range(episodes_per_seed):
            print(f"Seed {seed} | Episode {episode + 1}/{episodes_per_seed}")
            agent_rewards, events, step_count = _run_episode(
                tf_env, base_py_env, environment, act, map_size, heatmaps, behavior_totals, render, render_delay
            )
            team_reward = float(np.sum(agent_rewards))
            print(f"Seed {seed} | Episode {episode + 1} finished. Total reward: {team_reward:.2f}, Steps: {step_count}")
            seed_agent_rewards.append(agent_rewards)
            all_agent_rewards.append(agent_rewards)
            for name, value in events.items():
                seed_events[name].append(value)
            episodes.append({
                'seed': seed,
                'episode': episode + 1,
                'reward': team_reward,
                'agent_a_reward': float(agent_rewards[0]),
                'agent_b_reward': float(agent_rewards[1]),
                'steps': step_count,
                'behavior_events': events,
            })
        seed_summaries.append({
            'seed': seed,
            'episodes': len(seed_agent_rewards),
            **summarize_episode_rewards(np.asarray(seed_agent_rewards)),
            'behavior_events': summarize_behavior_events(seed_events),
        })
        tf_env.close()

    pooled = summarize_episode_rewards(np.asarray(all_agent_rewards))
    seed_means = [summary['mean_reward'] for summary in seed_summaries]
    results = {
        'experiment_id': metadata['experiment_id'],
        'checkpoint_step': int(train_step_counter.numpy()),
        'grid_size': map_size,
        'deterministic': deterministic,
        'seeds': seeds,
        'episodes_per_seed': episodes_per_seed,
        **pooled,
        'mean_reward_across_seeds': float(np.mean(seed_means)),
        'std_reward_across_seeds': float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        'by_seed': seed_summaries,
        'episodes': episodes,
        'behavior': {
            'map_size': map_size,
            'events_total': behavior_totals,
            'events_by_evaluation_seed': [
                {'seed': entry['seed'], **entry['behavior_events']} for entry in seed_summaries
            ],
            'total_stags_caught': int(behavior_totals.get('stags_caught_agent_a', 0)
                                      + behavior_totals.get('stags_caught_agent_b', 0)),
            'total_plants_eaten': int(behavior_totals.get('plants_eaten_agent_a', 0)
                                      + behavior_totals.get('plants_eaten_agent_b', 0)),
            'total_maulings_sustained': int(behavior_totals.get('maulings_sustained_agent_a', 0)
                                           + behavior_totals.get('maulings_sustained_agent_b', 0)),
            'total_plants_harvested_young': int(behavior_totals.get('plants_harvested_young', 0)),
            'total_plants_harvested_mature': int(behavior_totals.get('plants_harvested_mature', 0)),
            'heatmap_agent_a': heatmaps[0].tolist(),
            'heatmap_agent_b': heatmaps[1].tolist(),
        },
    }
    (experiment_dir / output_filename).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f"Mean reward: {pooled['mean_reward']:.4f}")
    print(f"Standard deviation: {pooled['std_reward']:.4f}")
    print(f"Agent A mean/std: {pooled['mean_agent_a_reward']:.4f} / {pooled['std_agent_a_reward']:.4f}")
    print(f"Agent B mean/std: {pooled['mean_agent_b_reward']:.4f} / {pooled['std_agent_b_reward']:.4f}")
    print("Evaluation complete.")
    return results
