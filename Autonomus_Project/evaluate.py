import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
import numpy as np
import tensorflow as tf

from tf_agents.environments import wrappers, tf_py_environment
from tf_agents.utils import common

# Import our modules.
from envs.tf_wrapper import TFStagHuntWrapper
from agents.networks import ActorNetwork

# =========================================================
# Wrapper (must remain identical to train.py).
# =========================================================
class TitaniumWrapper(wrappers.PyEnvironmentBaseWrapper):
    def _step(self, action):
        return self._sanitize(self._env.step(action))
    def _reset(self):
        return self._sanitize(self._env.reset())
    def _sanitize(self, ts):
        obs = np.asarray(ts.observation, dtype=np.float32)
        rew = np.asarray(ts.reward, dtype=np.float32).reshape(-1)
        disc = np.asarray(ts.discount, dtype=np.float32)
        return ts._replace(observation=obs, reward=rew, discount=disc)


def evaluate_agent():
    print("1. Initializing visual environment...")
    # Enable hard mode to test the learned policy under real conditions.
    py_env = TFStagHuntWrapper(render_mode="human", run_away_after_maul=False)
    py_env = TitaniumWrapper(py_env)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    print("2. Rebuilding the actor network only...")
    # Recreate the exact network used by train.py.
    actor_net = ActorNetwork(
        input_tensor_spec=tf_env.observation_spec(),
        output_tensor_spec=tf_env.action_spec(),
        hidden_sizes=(128, 128)
    )

    print("3. Loading checkpoint state...")
    checkpoint_dir = './checkpoints'
    train_step_counter = tf.Variable(0, dtype=tf.int64)

    # The checkpoint key must match the one used by train.py.
    checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        actor_network=actor_net,
        global_step=train_step_counter
    )

    # Load the weights.
    status = checkpointer.initialize_or_restore()
    # expect_partial() allows the critic and optimizer to be absent.
    status.expect_partial() 
    
    print(f"--> Modello ricaricato con successo dallo Step: {train_step_counter.numpy()}")
    time.sleep(3)

    print("4. Starting evaluation...")
    num_episodes = 1 
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        time_step = tf_env.reset()
        episode_reward = 0.0
        step_count = 0
        max_steps = 100  # Increase if the game requires more moves.
        
        while not time_step.is_last()[0] and step_count < max_steps:
            step_count += 1
            
            # Request an action from the network.
            action_distribution, _ = actor_net(time_step.observation)
            
            # Sample instead of using argmax to preserve some agent flexibility.
            raw_action = action_distribution.sample()
            action = tf.cast(raw_action, tf.int32)
            
            # Execute the move.
            time_step = tf_env.step(action)
            step_reward = float(tf.reduce_sum(time_step.reward))
            episode_reward += step_reward

            # Clear the console and render.
            os.system('cls' if os.name == 'nt' else 'clear')
            py_env.env.render()
            
            print(f"Step: {step_count} | Selected actions: {action.numpy()} | Immediate reward: {step_reward}")
            time.sleep(0.3)
            
        # Render the final state.
        os.system('cls' if os.name == 'nt' else 'clear')
        py_env.env.render()
        print(f"Episode {episode + 1} finished. Total reward: {episode_reward:.2f}, Steps: {step_count}")

    print("Evaluation complete. You can close the game window.")

if __name__ == "__main__":
    evaluate_agent()