"""On-policy rollout storage backed by a TF-Agents uniform replay buffer."""
from tf_agents.policies import actor_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory


class MemoryManager:
    def __init__(self, tf_env, agent, max_length: int = 260):
        """Allocates the on-policy buffer matching the environment's trajectory contract."""
        self.tf_env = tf_env
        self.agent = agent

        # Parallel workers expose both agents as one vector action, which is
        # not representable by TF-Agents' scalar ActorPolicy.
        if len(self.tf_env.observation_spec().shape) == 2:
            time_step_spec = self.tf_env.time_step_spec()
            self.trajectory_spec = trajectory.Trajectory(
                step_type=time_step_spec.step_type,
                observation=time_step_spec.observation,
                action=self.tf_env.action_spec(),
                policy_info=(),
                next_step_type=time_step_spec.step_type,
                reward=time_step_spec.reward,
                discount=time_step_spec.discount,
            )
        else:
            self.training_policy = actor_policy.ActorPolicy(
                time_step_spec=self.tf_env.time_step_spec(),
                action_spec=self.tf_env.action_spec(),
                actor_network=self.agent.actor,
            )
            self.trajectory_spec = self.training_policy.trajectory_spec

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.trajectory_spec,
            batch_size=self.tf_env.batch_size,
            max_length=max_length,
        )

    def get_sequential_data(self):
        """Returns all trajectories currently stored, in temporal order."""
        return self.replay_buffer.gather_all()

    def clear_buffer(self):
        """Removes collected trajectories before the next rollout."""
        self.replay_buffer.clear()
