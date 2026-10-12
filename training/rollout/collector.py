"""Experience collection: one on-policy rollout plus behaviour and movement statistics."""
import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep


# With a global seed set, every eager stateful random op gets a fresh op seed
# and TensorFlow caches a new kernel per call; tracing once fixes the seed.
@tf.function
def sample_actions(actor, observations):
    """Samples integer actions from an actor's categorical policy."""
    action_distribution, _ = actor(observations)
    return tf.cast(action_distribution.sample(), tf.int32)


def select_actions(agent, observation):
    """Chooses one action per agent with the shared policy or one policy per agent."""
    if agent.architecture == 'shared_shared':
        return sample_actions(agent.actor, observation)
    if observation.shape.rank == 2:
        actions = [
            sample_actions(agent.actor_for(agent_index), observation[agent_index:agent_index + 1])[0]
            for agent_index in range(2)
        ]
        return tf.stack(actions)
    actions = [
        sample_actions(agent.actor_for(agent_index), observation[:, agent_index, :])
        for agent_index in range(2)
    ]
    return tf.stack(actions, axis=1)


def new_rollout_stats():
    """Empty per-rollout behaviour counters."""
    return {
        'stags': 0,
        'plants': 0,
        'mauling': 0,
        'action_counts': np.zeros((2, 5), dtype=int),
        'stag_distance_sum': np.zeros(2, dtype=float),
        'observation_count': 0,
    }


def _record_positions(observations, map_size, stats, heatmap_a, heatmap_b):
    """Accumulates stag distances and visited cells from a batch of observations."""
    if observations.ndim == 2:
        observations = observations[None, ...]
    for environment_observation in observations:
        stag_position = environment_observation[0, 4:6]
        for agent_index in range(2):
            stats['stag_distance_sum'][agent_index] += float(
                np.abs(environment_observation[agent_index, 0:2] - stag_position).sum()
            )
        stats['observation_count'] += 1
        xa = min(int(environment_observation[0, 0]), map_size - 1)
        ya = min(int(environment_observation[0, 1]), map_size - 1)
        xb = min(int(environment_observation[1, 0]), map_size - 1)
        yb = min(int(environment_observation[1, 1]), map_size - 1)
        heatmap_a[xa, ya] += 1
        heatmap_b[xb, yb] += 1


def _record_reward_events(rewards, stats):
    """Attributes team rewards of one step to Hunt game events (reward-magnitude heuristic)."""
    if rewards.ndim == 1:
        rewards = rewards[None, ...]
    team_rewards = np.sum(rewards, axis=1)
    stats['stags'] += int(np.count_nonzero(team_rewards >= 10.0))
    stats['plants'] += int(np.count_nonzero((team_rewards >= 1.0) & (team_rewards <= 2.0)))
    stats['mauling'] += int(np.count_nonzero(team_rewards < 0.0))


def collect_rollout(agent, tf_env, memory, time_step, config):
    """Collects `config.rollout_steps` transitions; returns (time_step, heatmap_a, heatmap_b, stats)."""
    heatmap_a = np.zeros((config.map_size, config.map_size), dtype=int)
    heatmap_b = np.zeros((config.map_size, config.map_size), dtype=int)
    stats = new_rollout_stats()

    for _ in range(config.rollout_steps):
        action_tensor = select_actions(agent, time_step.observation)
        action_step = PolicyStep(action=action_tensor)

        next_time_step = tf_env.step(action_step.action)
        memory.replay_buffer.add_batch(
            trajectory.from_transition(time_step, action_step, next_time_step)
        )

        actions_taken = action_tensor.numpy().reshape(-1, 2)
        for agent_index in range(2):
            stats['action_counts'][agent_index] += np.bincount(
                actions_taken[:, agent_index], minlength=5
            )[:5]
        _record_positions(next_time_step.observation.numpy(), config.map_size, stats, heatmap_a, heatmap_b)
        _record_reward_events(time_step.reward.numpy(), stats)

        time_step = next_time_step

    return time_step, heatmap_a, heatmap_b, stats
