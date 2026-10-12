"""One PPO update from a collected rollout: reward mixing, GAE and minibatch epochs."""
import tensorflow as tf


@tf.function
def shuffled_indices(count):
    """Returns a random permutation of minibatch sample indices."""
    return tf.random.shuffle(tf.range(count))


def mix_agent_rewards(rewards, sharing_coef):
    """Blends each agent's reward with its partner's: r_i' = (1-c) r_i + c r_j."""
    if not 0.0 <= sharing_coef <= 1.0:
        raise ValueError('reward_sharing_coef must be between 0.0 and 1.0.')
    if rewards.shape.rank is not None and rewards.shape[-1] != 2:
        raise ValueError('Reward sharing currently supports exactly two agents.')
    coefficient = tf.cast(sharing_coef, rewards.dtype)
    return (1.0 - coefficient) * rewards + coefficient * tf.reverse(rewards, axis=[-1])


def critic_values(agent, trainer_index, states):
    """Evaluates states with a critic and restores the original value scale."""
    values, _ = agent.critic_for(trainer_index)(states)
    return agent.trainers[trainer_index].denormalize_values(values)


def generalized_advantages(rewards, old_values, next_values, dones, gamma, lam):
    """GAE(gamma, lambda) along the time axis (axis 1); returns (advantages, returns)."""
    deltas = rewards + gamma * next_values * (1.0 - dones) - old_values
    gae = tf.zeros_like(rewards[:, 0])
    advantages_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(rewards)[1])
    for step_index in tf.range(tf.shape(rewards)[1] - 1, -1, -1):
        gae = deltas[:, step_index] + gamma * lam * (1.0 - dones[:, step_index]) * gae
        advantages_ta = advantages_ta.write(step_index, gae)
    stacked = advantages_ta.stack()
    # TensorArray stacks time first; restore [environment, time, (agent)].
    advantages = tf.transpose(stacked, [1, 0, 2]) if stacked.shape.rank == 3 else tf.transpose(stacked)
    return advantages, advantages + old_values


def standardize(advantages):
    """Zero-mean, unit-variance advantages over the whole rollout."""
    return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)


def _minibatch_updates(agent, trainer_index, config, global_step, states, actions, old_log_probs,
                       advantages, returns, old_values, critic_states=None, critic_output_index=None):
    """Shuffled minibatch PPO epochs over flattened samples for one trainer."""
    flat = {
        'states': tf.reshape(states, [-1, tf.shape(states)[-1]]),
        'actions': tf.reshape(actions, [-1]),
        'old_log_probs': tf.reshape(old_log_probs, [-1]),
        'advantages': tf.reshape(advantages, [-1]),
        'returns': tf.reshape(returns, [-1]),
        'old_values': tf.reshape(old_values, [-1]),
    }
    if critic_states is not None:
        flat['critic_states'] = tf.reshape(critic_states, [-1, tf.shape(critic_states)[-1]])
    extra = {} if critic_output_index is None else {'critic_output_index': critic_output_index}

    sample_count = tf.shape(flat['states'])[0]
    minibatch_starts = tf.range(0, sample_count, config.minibatch_size)
    train_loss, diagnostics = 0.0, {}
    for _ in range(config.ppo_epochs):
        permutation = shuffled_indices(sample_count)
        for start_index in minibatch_starts:
            indices = permutation[start_index:start_index + config.minibatch_size]
            train_loss, diagnostics = agent.train_step_for(
                trainer_index,
                **{name: tf.gather(tensor, indices) for name, tensor in flat.items()},
                **extra,
                global_step=tf.cast(global_step, tf.int64),
            )
    return train_loss, diagnostics


def _update_joint_critic(agent, config, global_step, s_t, s_t_next, a_t, r_t, dones):
    """Centralized critic over both observations, one value head per agent."""
    joint_states = tf.concat([s_t[:, :, 0, :], s_t[:, :, 1, :]], axis=-1)
    joint_next_states = tf.concat([s_t_next[:, :, 0, :], s_t_next[:, :, 1, :]], axis=-1)
    old_values = tf.reshape(critic_values(agent, 0, joint_states), tf.shape(r_t))
    next_values = tf.reshape(critic_values(agent, 0, joint_next_states), tf.shape(r_t))
    advantages, returns = generalized_advantages(
        r_t, old_values, next_values, dones, config.gamma, config.gae_lambda
    )
    agent.update_value_normalizer(returns)
    advantages = standardize(advantages)

    train_loss, diagnostics = 0.0, {}
    for agent_index in range(2):
        trainer_index = 0 if agent.architecture == 'shared_shared' else agent_index
        agent_states = s_t[:, :, agent_index, :]
        old_distribution, _ = agent.actor_for(trainer_index)(agent_states)
        old_log_probs = old_distribution.log_prob(tf.cast(a_t[:, :, agent_index], tf.int32))
        train_loss, diagnostics = _minibatch_updates(
            agent, trainer_index, config, global_step,
            agent_states, a_t[:, :, agent_index], old_log_probs,
            advantages[:, :, agent_index], returns[:, :, agent_index], old_values[:, :, agent_index],
            critic_states=joint_states, critic_output_index=agent_index,
        )
    return train_loss, diagnostics


def _update_shared_local(agent, config, global_step, s_t, s_t_next, a_t, r_t, dones):
    """One shared actor-critic for both agents; minibatches are drawn over time steps."""
    old_values = tf.reshape(critic_values(agent, 0, s_t), tf.shape(r_t))
    next_values = tf.reshape(critic_values(agent, 0, s_t_next), tf.shape(r_t))
    advantages, returns = generalized_advantages(
        r_t, old_values, next_values, dones, config.gamma, config.gae_lambda
    )
    agent.update_value_normalizer(returns)
    advantages = standardize(advantages)
    old_distribution, _ = agent.actor(s_t)
    old_log_probs = old_distribution.log_prob(tf.cast(a_t, tf.int32))

    time_steps = tf.shape(r_t)[1]
    minibatch_starts = tf.range(0, time_steps, config.minibatch_size)
    train_loss, diagnostics = 0.0, {}
    for _ in range(config.ppo_epochs):
        permutation = shuffled_indices(time_steps)
        for start_index in minibatch_starts:
            time_indices = permutation[start_index:start_index + config.minibatch_size]
            train_loss, diagnostics = agent.train_step(
                states=tf.reshape(tf.gather(s_t, time_indices, axis=1), [-1, tf.shape(s_t)[-1]]),
                actions=tf.reshape(tf.gather(a_t, time_indices, axis=1), [-1]),
                old_log_probs=tf.reshape(tf.gather(old_log_probs, time_indices, axis=1), [-1]),
                advantages=tf.reshape(tf.gather(advantages, time_indices, axis=1), [-1]),
                returns=tf.reshape(tf.gather(returns, time_indices, axis=1), [-1]),
                old_values=tf.reshape(tf.gather(old_values, time_indices, axis=1), [-1]),
                global_step=tf.cast(global_step, tf.int64),
            )
    return train_loss, diagnostics


def _update_per_agent(agent, config, global_step, s_t, s_t_next, a_t, r_t, dones):
    """Independent GAE and updates for each agent's own (actor, critic) trainer."""
    train_loss, diagnostics = 0.0, {}
    for agent_index in range(2):
        agent_states = s_t[:, :, agent_index, :]
        agent_rewards = r_t[:, :, agent_index]
        old_values = tf.reshape(
            critic_values(agent, agent_index, agent_states), tf.shape(agent_rewards)
        )
        next_values = tf.reshape(
            critic_values(agent, agent_index, s_t_next[:, :, agent_index, :]), tf.shape(agent_rewards)
        )
        advantages, returns = generalized_advantages(
            agent_rewards, old_values, next_values, dones[:, :, agent_index],
            config.gamma, config.gae_lambda,
        )
        agent.update_value_normalizer(returns, agent_index)
        advantages = standardize(advantages)
        old_distribution, _ = agent.actor_for(agent_index)(agent_states)
        old_log_probs = old_distribution.log_prob(tf.cast(a_t[:, :, agent_index], tf.int32))
        train_loss, diagnostics = _minibatch_updates(
            agent, agent_index, config, global_step,
            agent_states, a_t[:, :, agent_index], old_log_probs, advantages, returns, old_values,
        )
    return train_loss, diagnostics


def train_ppo_epochs(agent, trajectories, current_global_step, config):
    """Runs the PPO update for one rollout; returns (loss, team_return, agent_returns, diagnostics)."""
    states = trajectories.observation
    actions = trajectories.action
    rewards = trajectories.reward
    next_step_types = trajectories.next_step_type

    # Normalize the legacy [agent, time, feature] buffer to [environment, time, agent, feature].
    if states.shape.rank == 3:
        states = tf.transpose(states, [1, 0, 2])[None, ...]
        actions = tf.transpose(actions, [1, 0])[None, ...]
        rewards = tf.transpose(rewards, [1, 0])[None, ...]
        next_step_types = next_step_types[0:1, :]

    s_t, s_t_next = states[:, :-1, :], states[:, 1:, :]
    a_t = actions[:, :-1]
    raw_r_t = rewards[:, :-1]
    r_t = mix_agent_rewards(raw_r_t, config.reward_sharing_coef)
    agent_returns = tf.reduce_sum(raw_r_t, axis=1)
    custom_return = tf.reduce_mean(agent_returns)

    done_mask = tf.cast(next_step_types[:, :-1] == 2, tf.float32)
    dones = done_mask[:, :, None] * tf.ones_like(r_t)

    if config.critic_observation == 'joint':
        update = _update_joint_critic
    elif agent.architecture == 'shared_shared':
        update = _update_shared_local
    else:
        update = _update_per_agent
    train_loss, diagnostics = update(agent, config, current_global_step, s_t, s_t_next, a_t, r_t, dones)
    return train_loss, custom_return, agent_returns, diagnostics
