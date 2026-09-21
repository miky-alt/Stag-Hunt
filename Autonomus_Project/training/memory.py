from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import actor_policy # Add the actor policy import.

class MemoryManager:
    def __init__(self, tf_env, agent, max_length: int = 65):
        """
        Initializes the memory components for ON-POLICY training.
        """
        self.tf_env = tf_env
        self.agent = agent
        
        # 1. Create the exploration policy here so it becomes the system's core.
        self.training_policy = actor_policy.ActorPolicy(
            time_step_spec=self.tf_env.time_step_spec(),
            action_spec=self.tf_env.action_spec(),
            actor_network=self.agent.actor # The configured actor network.
        )

        # 2. Initialize the rollout buffer from the policy specification.
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.training_policy.trajectory_spec, # Critical specification.
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