import numpy as np
import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep


def _critic_values(agent, trainer_index, states):
    values, _ = agent.critic_for(trainer_index)(states)
    return agent.trainers[trainer_index].denormalize_values(values)


def _mix_agent_rewards(rewards, sharing_coef):
    """Mix each agent's reward with the other agent's reward for training."""
    if not 0.0 <= sharing_coef <= 1.0:
        raise ValueError('reward_sharing_coef must be between 0.0 and 1.0.')
    if rewards.shape.rank is not None and rewards.shape[-1] != 2:
        raise ValueError('Reward sharing currently supports exactly two agents.')
    coefficient = tf.cast(sharing_coef, rewards.dtype)
    return (1.0 - coefficient) * rewards + coefficient * tf.reverse(rewards, axis=[-1])


def collect_rollout(agent, tf_env, memory, time_step, config):
    """Collect one rollout and return environment statistics."""
    fast_heatmap_a = np.zeros((config.map_size, config.map_size), dtype=int)
    fast_heatmap_b = np.zeros((config.map_size, config.map_size), dtype=int)
    stats = {'stags': 0, 'plants': 0, 'mauling': 0}

    for _ in range(config.rollout_steps):
        if agent.architecture == 'shared_shared':
            action_distribution, _ = agent.actor(time_step.observation)
            action_tensor = tf.cast(action_distribution.sample(), tf.int32)
        else:
            observations = time_step.observation
            if observations.shape.rank == 2:
                actions = [
                    agent.actor_for(agent_index)(
                        observations[agent_index:agent_index + 1]
                    )[0].sample()[0]
                    for agent_index in range(2)
                ]
                action_tensor = tf.stack(actions)
            else:
                actions = [
                    agent.actor_for(agent_index)(
                        observations[:, agent_index, :]
                    )[0].sample()
                    for agent_index in range(2)
                ]
                action_tensor = tf.stack(actions, axis=1)
        action_step = PolicyStep(action=action_tensor)

        next_time_step = tf_env.step(action_step.action)
        memory.replay_buffer.add_batch(
            trajectory.from_transition(time_step, action_step, next_time_step)
        )

        observations = next_time_step.observation.numpy()
        if observations.ndim == 2:
            observations = observations[None, ...]
        for environment_observation in observations:
            xa, ya = (
                min(int(environment_observation[0, 0]), config.map_size - 1),
                min(int(environment_observation[0, 1]), config.map_size - 1),
            )
            xb, yb = (
                min(int(environment_observation[1, 0]), config.map_size - 1),
                min(int(environment_observation[1, 1]), config.map_size - 1),
            )
            fast_heatmap_a[xa, ya] += 1
            fast_heatmap_b[xb, yb] += 1

        previous_rewards = time_step.reward.numpy()
        if previous_rewards.ndim == 1:
            previous_rewards = previous_rewards[None, ...]
        episode_rewards = np.sum(previous_rewards, axis=1)
        stats['stags'] += int(np.count_nonzero(episode_rewards >= 10.0))
        stats['plants'] += int(np.count_nonzero(
            (episode_rewards >= 1.0) & (episode_rewards <= 2.0)
        ))
        stats['mauling'] += int(np.count_nonzero(episode_rewards < 0.0))

        time_step = next_time_step

    return time_step, fast_heatmap_a, fast_heatmap_b, stats


def train_ppo_epochs(agent, trajectories, current_global_step, config):
    """Prepare one rollout and perform shuffled minibatch PPO updates."""
    states = trajectories.observation
    actions = trajectories.action
    rewards = trajectories.reward
    next_step_types = trajectories.next_step_type

    # Normalize the legacy [agent, time, feature] buffer to
    # [environment, time, agent, feature].
    if states.shape.rank == 3:
        states = tf.transpose(states, [1, 0, 2])[None, ...]
        actions = tf.transpose(actions, [1, 0])[None, ...]
        rewards = tf.transpose(rewards, [1, 0])[None, ...]
        next_step_types = next_step_types[0:1, :]

    s_t = states[:, :-1, :]
    s_t_next = states[:, 1:, :]
    a_t = actions[:, :-1]
    raw_r_t = rewards[:, :-1]
    r_t = _mix_agent_rewards(raw_r_t, config.reward_sharing_coef)
    agent_returns = tf.reduce_sum(raw_r_t, axis=1)
    custom_return = tf.reduce_mean(agent_returns)

    next_step_t = next_step_types[:, :-1]
    done_mask = tf.cast(next_step_t == 2, tf.float32)
    dones = done_mask[:, :, None] * tf.ones_like(r_t)

    gamma = config.gamma
    lam = config.gae_lambda
    if config.critic_observation == 'joint':
        joint_states = tf.concat([s_t[:, :, 0, :], s_t[:, :, 1, :]], axis=-1)
        joint_next_states = tf.concat(
            [s_t_next[:, :, 0, :], s_t_next[:, :, 1, :]], axis=-1
        )
        old_values = _critic_values(agent, 0, joint_states)
        next_values = _critic_values(agent, 0, joint_next_states)
        old_values = tf.reshape(old_values, tf.shape(r_t))
        next_values = tf.reshape(next_values, tf.shape(r_t))
        deltas = r_t + gamma * next_values * (1.0 - dones) - old_values
        gae = tf.zeros_like(r_t[:, 0])
        advantages_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(r_t)[1])
        for step_index in tf.range(tf.shape(r_t)[1] - 1, -1, -1):
            gae = deltas[:, step_index] + gamma * lam * (1.0 - dones[:, step_index]) * gae
            advantages_ta = advantages_ta.write(step_index, gae)
        advantages = tf.transpose(advantages_ta.stack(), [1, 0, 2])
        returns = advantages + old_values
        agent.update_value_normalizer(returns)
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + 1e-8
        )

        train_loss = 0.0
        diagnostics = {}
        for agent_index in range(2):
            actor = agent.actor_for(0 if agent.architecture == 'shared_shared' else agent_index)
            agent_states = s_t[:, :, agent_index, :]
            old_distribution, _ = actor(agent_states)
            old_log_probs = old_distribution.log_prob(
                tf.cast(a_t[:, :, agent_index], tf.int32)
            )
            flat_states = tf.reshape(agent_states, [-1, tf.shape(agent_states)[-1]])
            flat_joint_states = tf.reshape(
                joint_states, [-1, tf.shape(joint_states)[-1]]
            )
            flat_actions = tf.reshape(a_t[:, :, agent_index], [-1])
            flat_old_log_probs = tf.reshape(old_log_probs, [-1])
            flat_advantages = tf.reshape(advantages[:, :, agent_index], [-1])
            flat_returns = tf.reshape(returns[:, :, agent_index], [-1])
            sample_count = tf.shape(flat_states)[0]
            minibatch_starts = tf.range(0, sample_count, config.minibatch_size)
            trainer_index = 0 if agent.architecture == 'shared_shared' else agent_index
            for _ in range(config.ppo_epochs):
                shuffled_indices = tf.random.shuffle(tf.range(sample_count))
                for start_index in minibatch_starts:
                    indices = shuffled_indices[start_index:start_index + config.minibatch_size]
                    train_loss, diagnostics = agent.train_step_for(
                        trainer_index,
                        states=tf.gather(flat_states, indices),
                        actions=tf.gather(flat_actions, indices),
                        old_log_probs=tf.gather(flat_old_log_probs, indices),
                        advantages=tf.gather(flat_advantages, indices),
                        returns=tf.gather(flat_returns, indices),
                        old_values=tf.gather(
                            tf.reshape(old_values[:, :, agent_index], [-1]), indices
                        ),
                        critic_states=tf.gather(flat_joint_states, indices),
                        critic_output_index=agent_index,
                        global_step=tf.cast(current_global_step, tf.int64),
                    )
    elif agent.architecture == 'shared_shared':
        old_values = _critic_values(agent, 0, s_t)
        old_values = tf.reshape(old_values, tf.shape(r_t))
        next_values = _critic_values(agent, 0, s_t_next)
        next_values = tf.reshape(next_values, tf.shape(r_t))
        deltas = r_t + gamma * next_values * (1.0 - dones) - old_values
        gae = tf.zeros_like(r_t[:, 0])
        advantages_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(r_t)[1])
        for step_index in tf.range(tf.shape(r_t)[1] - 1, -1, -1):
            gae = deltas[:, step_index] + gamma * lam * (1.0 - dones[:, step_index]) * gae
            advantages_ta = advantages_ta.write(step_index, gae)
        advantages = tf.transpose(advantages_ta.stack(), [1, 0, 2])
        returns = advantages + old_values
        agent.update_value_normalizer(returns)

        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        old_distribution, _ = agent.actor(s_t)
        old_log_probs = old_distribution.log_prob(tf.cast(a_t, tf.int32))
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
                    old_values=tf.reshape(tf.gather(old_values, time_indices, axis=1), [-1]),
                    global_step=tf.cast(current_global_step, tf.int64),
                )
    else:
        train_loss = 0.0
        diagnostics = {}
        for agent_index in range(2):
            agent_states = s_t[:, :, agent_index, :]
            agent_next_states = s_t_next[:, :, agent_index, :]
            agent_rewards = r_t[:, :, agent_index]
            agent_dones = dones[:, :, agent_index]
            old_values = _critic_values(agent, agent_index, agent_states)
            next_values = _critic_values(agent, agent_index, agent_next_states)
            old_values = tf.reshape(old_values, tf.shape(agent_rewards))
            next_values = tf.reshape(next_values, tf.shape(agent_rewards))
            deltas = agent_rewards + gamma * next_values * (1.0 - agent_dones) - old_values
            gae = tf.zeros_like(agent_rewards[:, 0])
            advantages_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(agent_rewards)[1])
            for step_index in tf.range(tf.shape(agent_rewards)[1] - 1, -1, -1):
                gae = deltas[:, step_index] + gamma * lam * (1.0 - agent_dones[:, step_index]) * gae
                advantages_ta = advantages_ta.write(step_index, gae)
            advantages = tf.transpose(advantages_ta.stack())
            returns = advantages + old_values
            agent.update_value_normalizer(returns, agent_index)
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
            old_distribution, _ = agent.actor_for(agent_index)(agent_states)
            old_log_probs = old_distribution.log_prob(tf.cast(a_t[:, :, agent_index], tf.int32))

            flat_states = tf.reshape(agent_states, [-1, tf.shape(agent_states)[-1]])
            flat_actions = tf.reshape(a_t[:, :, agent_index], [-1])
            flat_old_log_probs = tf.reshape(old_log_probs, [-1])
            flat_advantages = tf.reshape(advantages, [-1])
            flat_returns = tf.reshape(returns, [-1])
            sample_count = tf.shape(flat_states)[0]
            minibatch_starts = tf.range(0, sample_count, config.minibatch_size)
            for _ in range(config.ppo_epochs):
                shuffled_indices = tf.random.shuffle(tf.range(sample_count))
                for start_index in minibatch_starts:
                    indices = shuffled_indices[start_index:start_index + config.minibatch_size]
                    train_loss, diagnostics = agent.train_step_for(
                        agent_index,
                        states=tf.gather(flat_states, indices),
                        actions=tf.gather(flat_actions, indices),
                        old_log_probs=tf.gather(flat_old_log_probs, indices),
                        advantages=tf.gather(flat_advantages, indices),
                        returns=tf.gather(flat_returns, indices),
                        old_values=tf.gather(tf.reshape(old_values, [-1]), indices),
                        global_step=tf.cast(current_global_step, tf.int64),
                    )

    return train_loss, custom_return, agent_returns, diagnostics
