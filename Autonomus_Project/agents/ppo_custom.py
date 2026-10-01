import tensorflow as tf


class ValueNormalizer:
    """Running mean and standard deviation for critic targets."""

    def __init__(self):
        self.mean = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.variance = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.count = tf.Variable(1.0, trainable=False, dtype=tf.float32)

    def update(self, values):
        values = tf.cast(tf.reshape(values, [-1]), tf.float32)
        batch_count = tf.cast(tf.size(values), tf.float32)
        batch_mean = tf.reduce_mean(values)
        batch_variance = tf.math.reduce_variance(values)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        new_variance = (
            self.variance * self.count
            + batch_variance * batch_count
            + tf.square(delta) * self.count * batch_count / total_count
        ) / total_count
        self.mean.assign(new_mean)
        self.variance.assign(tf.maximum(new_variance, 1e-8))
        self.count.assign(total_count)

    def normalize(self, values):
        return (values - self.mean) / tf.sqrt(self.variance + 1e-8)

    def denormalize(self, values):
        return values * tf.sqrt(self.variance + 1e-8) + self.mean


class MultiAgentPPO:
    """PPO container supporting shared or per-agent actor/critic networks."""

    VALID_ARCHITECTURES = {
        'shared_shared',
        'separate_actors_shared_critic',
        'separate_actors_separate_critics',
    }

    def __init__(self, trainers, architecture):
        if architecture not in self.VALID_ARCHITECTURES:
            raise ValueError(
                f'Unknown agent architecture: {architecture}. '
                f'Expected one of {sorted(self.VALID_ARCHITECTURES)}.'
            )
        self.trainers = trainers
        self.architecture = architecture

    @property
    def actor(self):
        return self.trainers[0].actor

    @property
    def critic(self):
        return self.trainers[0].critic

    @property
    def optimizer(self):
        return self.trainers[0].optimizer

    @property
    def actors(self):
        return [trainer.actor for trainer in self.trainers]

    @property
    def critics(self):
        return [trainer.critic for trainer in self.trainers]

    def actor_for(self, agent_index):
        return self.trainers[agent_index].actor

    def critic_for(self, agent_index):
        return self.trainers[agent_index].critic

    def optimizer_for(self, agent_index):
        return self.trainers[agent_index].optimizer

    def set_learning_rate(self, value):
        for trainer in self.trainers:
            trainer.set_learning_rate(value)

    @property
    def entropy_coef(self):
        return self.trainers[0].entropy_coef

    def train_step_for(self, agent_index, *args, **kwargs):
        return self.trainers[agent_index].train_step(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.trainers[0].train_step(*args, **kwargs)

    def update_value_normalizer(self, values, agent_index=0):
        self.trainers[agent_index].update_value_normalizer(values)


class CustomPPO:
    def __init__(
        self,
        actor_net,
        critic_net,
        lr=3e-4,
        clip_epsilon=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.02,
        value_normalization=False,
        value_clipping=False,
        value_clip_epsilon=0.2,
        value_normalizer=None,
    ):
        self.actor = actor_net
        self.critic = critic_net
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.value_clipping = value_clipping
        self.value_clip_epsilon = value_clip_epsilon
        self.value_normalizer = value_normalizer or (
            ValueNormalizer() if value_normalization else None
        )
        self.entropy_coef = tf.Variable(
            initial_value=entropy_coef, 
            dtype=tf.float32, 
            trainable=False, 
            name="entropy_coefficient"
        )

    def set_learning_rate(self, value):
        self.optimizer.learning_rate.assign(value)

    def update_value_normalizer(self, values):
        if self.value_normalizer is not None:
            self.value_normalizer.update(values)

    def denormalize_values(self, values):
        if self.value_normalizer is not None:
            return self.value_normalizer.denormalize(values)
        return values

    @tf.function
    def train_step(
        self,
        states,
        actions,
        old_log_probs,
        advantages,
        returns,
        global_step,
        critic_states=None,
        critic_output_index=None,
        old_values=None,
    ):
        """Apply one PPO gradient update to one minibatch."""
        if critic_states is None:
            critic_states = states
        with tf.GradientTape() as tape:
            # Recompute only what is needed for the current gradients.
            curr_distribution, _ = self.actor(states)
            curr_log_probs = curr_distribution.log_prob(tf.cast(actions, tf.int32))
            entropy = curr_distribution.entropy()

            # Current critic evaluation for differentiation.
            curr_values, _ = self.critic(critic_states)
            if critic_output_index is not None:
                curr_values = curr_values[..., critic_output_index]
            curr_values = tf.reshape(curr_values, tf.shape(returns))
            if self.value_normalizer is not None:
                normalized_returns = self.value_normalizer.normalize(returns)
                normalized_values = self.value_normalizer.normalize(curr_values)
                normalized_old_values = self.value_normalizer.normalize(
                    curr_values if old_values is None else old_values
                )
            else:
                normalized_returns = returns
                normalized_values = curr_values
                normalized_old_values = curr_values if old_values is None else old_values

            # TensorBoard logging.
            tf.summary.histogram('Actor/Softmax_Logits', curr_distribution.logits, step=global_step)
            tf.summary.histogram('Critic/Advantages', advantages, step=global_step)

            # Actor loss.
            ratios = tf.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            value_loss = tf.square(normalized_returns - normalized_values)
            if self.value_clipping:
                clipped_values = normalized_old_values + tf.clip_by_value(
                    normalized_values - normalized_old_values,
                    -self.value_clip_epsilon,
                    self.value_clip_epsilon,
                )
                clipped_value_loss = tf.square(normalized_returns - clipped_values)
                value_loss = tf.maximum(value_loss, clipped_value_loss)
            critic_loss = 0.5 * tf.reduce_mean(value_loss)
            
            # Entropy loss.
            entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy)

            # Total loss.
            total_loss = (
                actor_loss
                + self.value_loss_coef * critic_loss
                + entropy_loss
            )

            approx_kl = tf.reduce_mean(old_log_probs - curr_log_probs)
            clip_fraction = tf.reduce_mean(
                tf.cast(tf.abs(ratios - 1.0) > self.clip_epsilon, tf.float32)
            )

        # Compute gradients.
        trainable_variables = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        
        for grad, var in zip(gradients, trainable_variables):
            if grad is not None:
                clean_name = var.name.replace(':', '_')
                tf.summary.histogram(f'Gradients/{clean_name}', grad, step=global_step)

        # Global gradient clipping.
        gradients, global_grad_norm = tf.clip_by_global_norm(gradients, 0.5)
        
        # Apply gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        diagnostics = {
            'policy_loss': actor_loss,
            'value_loss': critic_loss,
            'entropy': tf.reduce_mean(entropy),
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'gradient_norm': global_grad_norm,
            'learning_rate': self.optimizer.learning_rate,
        }
        for name, value in diagnostics.items():
            tf.summary.scalar(f'PPO/{name}', value, step=global_step)

        return total_loss, diagnostics