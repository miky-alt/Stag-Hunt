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

    @tf.function
    def train_step(self, states, actions, old_log_probs, rewards, next_states, dones, global_step):
        gamma = 0.99
        lam = 0.95
        time_steps = tf.shape(rewards)[1]

        # --- 1. PRECOMPUTE GAE AND RETURNS (Outside GradientTape) ---
        # Old estimates do not require gradients.
        old_values, _ = self.critic(states)
        old_values = tf.reshape(old_values, tf.shape(rewards))
        
        next_values, _ = self.critic(next_states)
        next_values = tf.reshape(next_values, tf.shape(rewards))

        deltas = rewards + gamma * next_values * (1.0 - dones) - old_values
        
        gae = tf.zeros_like(rewards[:, 0])
        advantages_ta = tf.TensorArray(dtype=tf.float32, size=time_steps)
        
        for t in tf.range(time_steps - 1, -1, -1):
            gae = deltas[:, t] + gamma * lam * (1.0 - dones[:, t]) * gae
            advantages_ta = advantages_ta.write(t, gae)
            
        advantages = advantages_ta.stack()
        advantages = tf.transpose(advantages)
        
        # Static returns used to train the critic.
        returns = advantages + old_values

        # Normalize the advantages.
        adv_mean = tf.reduce_mean(advantages)
        adv_std = tf.math.reduce_std(advantages)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # --- 2. TRAINING AND BACKPROPAGATION ---
        with tf.GradientTape() as tape:
            # Recompute only what is needed for the current gradients.
            curr_distribution, _ = self.actor(states)
            curr_log_probs = curr_distribution.log_prob(tf.cast(actions, tf.int32))
            entropy = curr_distribution.entropy()

            # Current critic evaluation for differentiation.
            curr_values, _ = self.critic(states)
            curr_values = tf.reshape(curr_values, tf.shape(rewards))

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

        # Compute gradients.
        trainable_variables = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        
        for grad, var in zip(gradients, trainable_variables):
            if grad is not None:
                clean_name = var.name.replace(':', '_')
                tf.summary.histogram(f'Gradients/{clean_name}', grad, step=global_step)

        # Global gradient clipping.
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        
        # Apply gradients.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return total_loss