from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import actor_policy # Add the actor policy import.
from tf_agents.trajectories import trajectory

class MemoryManager:
    def __init__(self, tf_env, agent, max_length: int = 260):
        """
        Initializes the memory components for ON-POLICY training.
        """
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
                actor_network=self.agent.actor
            )
            self.trajectory_spec = self.training_policy.trajectory_spec

        # 2. Initialize the rollout buffer from the policy specification.
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.trajectory_spec,
            batch_size=self.tf_env.batch_size,
            max_length=max_length
        )

    def get_training_driver(self, collect_steps_per_iteration: int = 65, extra_observers=None):
            
            observers = [self.replay_buffer.add_batch]
            if extra_observers:
                observers.extend(extra_observers) # Add the extra observers.
                
            from tf_agents.drivers import dynamic_step_driver
            
            main_driver = dynamic_step_driver.DynamicStepDriver(
                env=self.tf_env,
                policy=self.training_policy, 
                observers=observers, # Use the updated observer list.
                num_steps=collect_steps_per_iteration
            )
            return main_driver
        
    def get_sequential_data(self):
        return self.replay_buffer.gather_all()

    def clear_buffer(self):
        self.replay_buffer.clear()