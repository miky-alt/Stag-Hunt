import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep


def collect_rollout(agent, tf_env, memory, time_step, config):
    """Collect one rollout and return environment statistics."""
    fast_heatmap_a = np.zeros((config.map_size, config.map_size), dtype=int)
    fast_heatmap_b = np.zeros((config.map_size, config.map_size), dtype=int)
    stats = {'stags': 0, 'plants': 0, 'mauling': 0}

    for _ in range(config.rollout_steps):
        action_distribution, _ = agent.actor(time_step.observation)
        action_tensor = tf.cast(action_distribution.sample(), tf.int32)
        action_step = PolicyStep(action=action_tensor)

        next_time_step = tf_env.step(action_step.action)
        memory.replay_buffer.add_batch(
            trajectory.from_transition(time_step, action_step, next_time_step)
        )

        flat_map = next_time_step.observation.numpy()[0]
        xa, ya = min(int(flat_map[0]), config.map_size - 1), min(int(flat_map[1]), config.map_size - 1)
        xb, yb = min(int(flat_map[2]), config.map_size - 1), min(int(flat_map[3]), config.map_size - 1)
        fast_heatmap_a[xa, ya] += 1
        fast_heatmap_b[xb, yb] += 1

        previous_reward = float(tf.reduce_sum(time_step.reward))
        if previous_reward >= 10.0:
            stats['stags'] += 1
        elif 1.0 <= previous_reward <= 2.0:
            stats['plants'] += 1
        elif previous_reward < 0.0:
            stats['mauling'] += 1

        time_step = next_time_step

    return time_step, fast_heatmap_a, fast_heatmap_b, stats


def train_ppo_epochs(agent, trajectories, current_global_step, config):
    """Prepare one rollout and perform shuffled minibatch PPO updates."""
    states = trajectories.observation
    actions = trajectories.action
    rewards = trajectories.reward
    next_step_types = trajectories.next_step_type

    s_t = states[:, :-1, :]
    s_t_next = states[:, 1:, :]
    a_t = actions[:, :-1]
    r_t = rewards[:, :-1]

    next_step_t = next_step_types[:, 1:]
    dones = tf.cast(tf.where(next_step_t == 2, tf.ones_like(r_t), tf.zeros_like(r_t)), tf.float32)

    old_values, _ = agent.critic(s_t)
    old_values = tf.reshape(old_values, tf.shape(r_t))
    next_values, _ = agent.critic(s_t_next)
    next_values = tf.reshape(next_values, tf.shape(r_t))

    gamma = 0.99
    lam = 0.95
    deltas = r_t + gamma * next_values * (1.0 - dones) - old_values
    gae = tf.zeros_like(r_t[:, 0])
    advantages_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(r_t)[1])
    for step_index in tf.range(tf.shape(r_t)[1] - 1, -1, -1):
        gae = deltas[:, step_index] + gamma * lam * (1.0 - dones[:, step_index]) * gae
        advantages_ta = advantages_ta.write(step_index, gae)
    advantages = tf.transpose(advantages_ta.stack())
    returns = advantages + old_values

    adv_mean = tf.reduce_mean(advantages)
    adv_std = tf.math.reduce_std(advantages)
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    old_distribution, _ = agent.actor(s_t)
    old_log_probs = old_distribution.log_prob(tf.cast(a_t, tf.int32))
    agent_returns = tf.reduce_sum(r_t, axis=1)
    custom_return = tf.reduce_mean(agent_returns)

    time_steps = tf.shape(r_t)[1]
    minibatch_starts = tf.range(0, time_steps, config.minibatch_size)
    train_loss = 0.0
    diagnostics = {}
    for _ in range(config.ppo_epochs):
        shuffled_indices = tf.random.shuffle(tf.range(time_steps))
        for start_index in minibatch_starts:
            time_indices = shuffled_indices[start_index:start_index + config.minibatch_size]
            train_loss, diagnostics = agent.train_step(
                states=tf.reshape(tf.gather(s_t, time_indices, axis=1), [-1, tf.shape(s_t)[-1]]),
                actions=tf.reshape(tf.gather(a_t, time_indices, axis=1), [-1]),
                old_log_probs=tf.reshape(tf.gather(old_log_probs, time_indices, axis=1), [-1]),
                advantages=tf.reshape(tf.gather(advantages, time_indices, axis=1), [-1]),
                returns=tf.reshape(tf.gather(returns, time_indices, axis=1), [-1]),
                global_step=tf.cast(current_global_step, tf.int64),
            )

    return train_loss, custom_return, agent_returns, diagnostics
