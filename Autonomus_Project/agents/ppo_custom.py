import tensorflow as tf

class CustomPPO:
    def __init__(self, actor_net, critic_net, lr=3e-4, clip_epsilon=0.2, entropy_coef=0.02):
        self.actor = actor_net
        self.critic = critic_net
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = tf.Variable(
            initial_value=entropy_coef, 
            dtype=tf.float32, 
            trainable=False, 
            name="entropy_coefficient"
        )

    def set_learning_rate(self, value):
        self.optimizer.learning_rate.assign(value)

    @tf.function
    def train_step(self, states, actions, old_log_probs, advantages, returns, global_step):
        """Apply one PPO gradient update to one minibatch."""
        with tf.GradientTape() as tape:
            # Recompute only what is needed for the current gradients.
            curr_distribution, _ = self.actor(states)
            curr_log_probs = curr_distribution.log_prob(tf.cast(actions, tf.int32))
            entropy = curr_distribution.entropy()

            # Current critic evaluation for differentiation.
            curr_values, _ = self.critic(states)
            curr_values = tf.reshape(curr_values, tf.shape(returns))

            # TensorBoard logging.
            tf.summary.histogram('Actor/Softmax_Logits', curr_distribution.logits, step=global_step)
            tf.summary.histogram('Critic/Advantages', advantages, step=global_step)

            # Actor loss.
            ratios = tf.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            # Critic loss (MSE on fixed returns).
            critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - curr_values))
            
            # Entropy loss.
            entropy_loss = -self.entropy_coef * tf.reduce_mean(entropy)

            # Total loss.
            total_loss = actor_loss + critic_loss + entropy_loss

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