"""TensorBoard logging for the training loop, kept out of the orchestration code."""
import numpy as np
import tensorflow as tf

from training.monitoring.heatmap import generate_annotated_heatmap

ACTION_NAMES = ('Left', 'Down', 'Right', 'Up', 'Stand')


def _variable_tag(variable):
    return variable.name.replace(':', '_').replace('/', '_')


class TensorBoardLogger:
    """Writes scalar, histogram and image summaries for one training run."""

    def __init__(self, log_dir, log_interval):
        self.writer = tf.summary.create_file_writer(str(log_dir))
        self.writer.set_as_default()
        self.log_interval = log_interval

    def log_iteration(self, step, metric, stats, diagnostics, agent_returns, heatmap_a, heatmap_b,
                      custom_return, learning_rate, entropy_coefficient, stable):
        """Scalars emitted every iteration."""
        tf.summary.scalar('Loss/Entropy_Coefficient', entropy_coefficient, step=step)
        tf.summary.scalar('Metrics/Custom_Average_Return', custom_return, step=step)
        tf.summary.scalar('Stability/Custom_Return_Window_Stable', float(bool(stable)), step=step)
        tf.summary.scalar('Metrics/Agent_A_Return', tf.reduce_mean(agent_returns[:, 0]), step=step)
        tf.summary.scalar('Metrics/Agent_B_Return', tf.reduce_mean(agent_returns[:, 1]), step=step)
        tf.summary.scalar('Training/Learning_Rate', learning_rate, step=step)
        tf.summary.scalar('Curriculum/Stag_Follow_Probability', metric['stag_follow_probability'], step=step)
        tf.summary.scalar('Behavior/Stags_Caught', stats['stags'], step=step)
        tf.summary.scalar('Behavior/Plants_Eaten', stats['plants'], step=step)
        tf.summary.scalar('Behavior/Maulings_Sustained', stats['mauling'], step=step)
        tf.summary.scalar('Policy/Entropy', diagnostics['entropy'], step=step)
        tf.summary.scalar('Policy/Approx_KL', diagnostics['approx_kl'], step=step)
        tf.summary.scalar('Policy/Clip_Fraction', diagnostics['clip_fraction'], step=step)

        observation_count = max(stats['observation_count'], 1)
        for agent_index, agent_label in enumerate(('A', 'B')):
            action_counts = stats['action_counts'][agent_index]
            action_total = max(int(action_counts.sum()), 1)
            for action_index, action_name in enumerate(ACTION_NAMES):
                tf.summary.scalar(
                    f'Actions/Agent_{agent_label}/{action_name}',
                    action_counts[action_index] / action_total, step=step,
                )
            tf.summary.scalar(
                f'Movement/Agent_{agent_label}_Stand_Fraction', action_counts[4] / action_total, step=step
            )
            tf.summary.scalar(
                f'Movement/Agent_{agent_label}_Mean_Stag_Distance',
                stats['stag_distance_sum'][agent_index] / observation_count, step=step,
            )
        tf.summary.scalar('Movement/Agent_A_Cells_Visited', int(np.count_nonzero(heatmap_a)), step=step)
        tf.summary.scalar('Movement/Agent_B_Cells_Visited', int(np.count_nonzero(heatmap_b)), step=step)

    def log_periodic(self, step, agent, diagnostics, global_heatmap_a, global_heatmap_b,
                     rollout_heatmap_a, rollout_heatmap_b):
        """Expensive histograms and heatmap images, only every `log_interval` steps."""
        if step % self.log_interval != 0:
            return
        for trainer_index, trainer in enumerate(agent.trainers):
            for variable in trainer.actor.trainable_variables:
                tf.summary.histogram(f'Networks/Actor_{trainer_index}/{_variable_tag(variable)}', variable, step=step)
            for variable in trainer.critic.trainable_variables:
                tf.summary.histogram(f'Networks/Critic_{trainer_index}/{_variable_tag(variable)}', variable, step=step)
            for gradient, variable in zip(
                diagnostics.get('gradients', []),
                trainer.actor.trainable_variables + trainer.critic.trainable_variables,
            ):
                if gradient is not None:
                    tf.summary.histogram(
                        f'Gradients/Trainer_{trainer_index}/{_variable_tag(variable)}', gradient, step=step
                    )
        tf.summary.image('Exploration/Heatmap_Agent_A', generate_annotated_heatmap(global_heatmap_a), step=step)
        tf.summary.image('Exploration/Heatmap_Agent_B', generate_annotated_heatmap(global_heatmap_b), step=step)
        tf.summary.image('Exploration/Rollout_Heatmap_Agent_A', generate_annotated_heatmap(rollout_heatmap_a), step=step)
        tf.summary.image('Exploration/Rollout_Heatmap_Agent_B', generate_annotated_heatmap(rollout_heatmap_b), step=step)

    def flush(self):
        self.writer.flush()
