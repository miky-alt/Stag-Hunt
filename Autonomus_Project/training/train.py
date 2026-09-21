import os
# Force compatibility with Keras 2 to avoid internal TensorFlow errors
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import gc
import time
import csv
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from dataclasses import dataclass

from tf_agents.environments import wrappers, tf_py_environment
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.trajectories.policy_step import PolicyStep

# Import our custom modules
from envs.tf_wrapper import TFStagHuntWrapper
from agents.networks import ActorNetwork, CriticNetwork
from agents.ppo_custom import CustomPPO
from training.memory import MemoryManager


# ==========================================
# 1. CONFIGURATION (Goodbye Magic Numbers)
# ==========================================
@dataclass
class TrainingConfig:
    """Stores all training hyperparameters and configuration values."""
    num_iterations: int = 16_000
    rollout_steps: int = 260
    ppo_epochs: int = 4
    curriculum_threshold: int = 3750
    log_interval: int = 200
    save_interval: int = 100
    map_size: int = 5
    
    # Agent hyperparameters
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    
    # Entropy
    entropy_start: float = 0.15
    entropy_end: float = 0.01
    entropy_decay_steps: int = 10000


# ==========================================
# 2. UTILITIES E CLASSI DI SUPPORTO
# ==========================================
class BlackBoxLogger:
    """Stores local episode logs and the grid's physical state data."""
    def __init__(self, log_dir="./logs_csv", map_size=(5, 5)):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_file = None
        self.writer = None
        self.map_size = map_size
        self.reset_episode_counters()

    def reset_episode_counters(self):
        self.heatmap_a = np.zeros(self.map_size, dtype=int)
        self.heatmap_b = np.zeros(self.map_size, dtype=int)
        self.stag_count = 0
        self.plant_count = 0
        self.mauling_count = 0
        self.event_snapshots = []

    def start_episode(self, episode_number):
        self.reset_episode_counters()
        path = os.path.join(self.log_dir, f"episode_{episode_number}.csv")
        self.current_file = open(path, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.current_file)
        self.writer.writerow([
            "Step", "Mossa_A", "Mossa_B", 
            "Pos_A", "Pos_B", "Pos_Stag", "Pos_Plant", 
            "Probs_A", "Probs_B", "Reward_Step", "Evento"
        ])

    def log_step(self, step_idx, actions, positions, probs_a, probs_b, reward):
        if not self.writer: return

        pos_a, pos_b = positions.get('A'), positions.get('B')
        
        if pos_a != "N/D" and isinstance(pos_a, (list, tuple, np.ndarray)) and len(pos_a) == 2:
            x, y = min(int(pos_a[0]), self.map_size[0]-1), min(int(pos_a[1]), self.map_size[1]-1)
            self.heatmap_a[x, y] += 1
            
        if pos_b != "N/D" and isinstance(pos_b, (list, tuple, np.ndarray)) and len(pos_b) == 2:
            x, y = min(int(pos_b[0]), self.map_size[0]-1), min(int(pos_b[1]), self.map_size[1]-1)
            self.heatmap_b[x, y] += 1

        evento_str = ""
        if reward == 10.0: 
            self.stag_count += 1
            evento_str = "CERVO CATTURATO"
            self.event_snapshots.append(f"Stag at Step {step_idx} | Pos A:{pos_a}, Pos B:{pos_b}, Stag:{positions.get('Stag')}")
        elif 1.0 <= reward <= 2.0:
            self.plant_count += 1
            evento_str = "PIANTA"
        elif reward < 0.0:
            self.mauling_count += 1
            evento_str = "MAULING (Cornata)"
            self.event_snapshots.append(f"Mauling at Step {step_idx} | Pos A:{pos_a}, Pos B:{pos_b}, Stag:{positions.get('Stag')}")

        self.writer.writerow([
            step_idx, actions[0], actions[1], pos_a, pos_b, 
            positions.get('Stag', 'N/D'), positions.get('Plant', 'N/D'), 
            probs_a, probs_b, reward, evento_str
        ])

    def close_episode(self):
        if self.current_file:
            self.writer.writerow([])
            self.writer.writerow(["--- RIASSUNTO COMPORTAMENTALE ---"])
            self.writer.writerow(["Total Stags Caught:", self.stag_count])
            self.writer.writerow(["Total Plants Eaten:", self.plant_count])
            self.writer.writerow(["Total Maulings Sustained:", self.mauling_count])
            self.writer.writerow([])
            self.writer.writerow(["--- SNAPSHOT EVENTI CRITICI ---"])
            for snapshot in self.event_snapshots: self.writer.writerow([snapshot])
            self.writer.writerow([])
            self.writer.writerow(["--- HEATMAP ESPLORAZIONE AGENTE A ---"])
            for row in self.heatmap_a: self.writer.writerow(row.tolist())
            self.writer.writerow([])
            self.writer.writerow(["--- HEATMAP ESPLORAZIONE AGENTE B ---"])
            for row in self.heatmap_b: self.writer.writerow(row.tolist())

            self.current_file.close()
            self.current_file = None


class EntropyScheduler:
    def __init__(self, agent, start_value=0.15, end_value=0.001, decay_steps=10000):
        self.agent = agent
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self, current_step):
        if current_step >= self.decay_steps:
            new_entropy = self.end_value
        else:
            completion_fraction = current_step / self.decay_steps
            new_entropy = self.start_value - completion_fraction * (self.start_value - self.end_value)
            
        self.agent.entropy_coef.assign(new_entropy)
        return new_entropy


def _generate_annotated_heatmap_tf(heatmap_counts):
    total = np.sum(heatmap_counts)
    percentage_grid = (heatmap_counts / total) * 100 if total > 0 else heatmap_counts
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(percentage_grid, cmap='Blues', vmin=0, vmax=100)
    
    for row in range(percentage_grid.shape[0]):
        for column in range(percentage_grid.shape[1]):
            value = percentage_grid[row, column]
            text_color = "white" if value > 45.0 else "black"
            ax.text(column, row, f"{value:.1f}%", va='center', ha='center', color=text_color, fontsize=9, weight='bold')
    
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    image_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(image_tensor, 0)


# ==========================================
# 3. ISOLATED TRAINING LOGIC (SRP)
# ==========================================
def _collect_rollout(agent, tf_env, memory, time_step, config: TrainingConfig):
    """Handles environment interaction and data collection exclusively (Phase A)."""
    fast_heatmap_a = np.zeros((config.map_size, config.map_size), dtype=int)
    fast_heatmap_b = np.zeros((config.map_size, config.map_size), dtype=int)
    stats = {'stags': 0, 'plants': 0, 'mauling': 0}

    for _ in range(config.rollout_steps):
        action_distribution, _ = agent.actor(time_step.observation)
        raw_action = action_distribution.sample()
        action_tensor = tf.cast(raw_action, tf.int32)
        action_step = PolicyStep(action=action_tensor)
        
        next_time_step = tf_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        memory.replay_buffer.add_batch(traj)
        
        # Fast heatmap update
        flat_map = next_time_step.observation.numpy()[0]
        xa, ya = min(int(flat_map[0]), config.map_size - 1), min(int(flat_map[1]), config.map_size - 1)
        xb, yb = min(int(flat_map[2]), config.map_size - 1), min(int(flat_map[3]), config.map_size - 1)
        
        fast_heatmap_a[xa, ya] += 1
        fast_heatmap_b[xb, yb] += 1

        # Counter update
        previous_reward = float(tf.reduce_sum(time_step.reward))
        if previous_reward >= 10.0: stats['stags'] += 1
        elif 1.0 <= previous_reward <= 2.0: stats['plants'] += 1
        elif previous_reward < 0.0: stats['mauling'] += 1

        time_step = next_time_step

    return time_step, fast_heatmap_a, fast_heatmap_b, stats


def _train_ppo_epochs(agent, trajectories, current_global_step, config: TrainingConfig):
    """Handles tensor preparation and backpropagation exclusively (Phases B/C)."""
    states = trajectories.observation
    actions = trajectories.action
    rewards = trajectories.reward
    next_step_types = trajectories.next_step_type
    
    s_t = states[:, :-1, :]       
    s_t_next = states[:, 1:, :]   
    a_t = actions[:, :-1]         
    r_t = rewards[:, :-1]         
    
    next_step_t = next_step_types[:, 1:] 
    dones = tf.where(next_step_t == 2, tf.ones_like(r_t), tf.zeros_like(r_t))
    dones = tf.cast(dones, tf.float32)

    old_distribution, _ = agent.actor(s_t)
    old_log_probs = old_distribution.log_prob(tf.cast(a_t, tf.int32))

    train_loss = 0.0
    for _ in range(config.ppo_epochs):
        train_loss = agent.train_step(
            states=s_t, actions=a_t, old_log_probs=old_log_probs, 
            rewards=r_t, next_states=s_t_next, dones=dones,
            global_step=tf.cast(current_global_step, tf.int64)
        )
        
    custom_return = tf.reduce_mean(tf.reduce_sum(r_t, axis=1))
    return train_loss, custom_return


# ==========================================
# 4. ORCHESTRATORE PRINCIPALE
# ==========================================
def train_agent():
    config = TrainingConfig()
    
    print("1. Initializing Environment...")
    logger_csv = BlackBoxLogger(map_size=(config.map_size, config.map_size))
    py_env = TFStagHuntWrapper(run_away_after_maul=True)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    print("2. Building Neural Networks...")
    actor_net = ActorNetwork(
        input_tensor_spec=tf_env.observation_spec(), 
        output_tensor_spec=tf_env.action_spec(),
        hidden_sizes=(128, 128)
    )
    value_net = CriticNetwork(
        input_tensor_spec=tf_env.observation_spec(),
        hidden_sizes=(128, 128)
    )

    print("3. Compiling the CUSTOM PPO Agent...")
    train_step_counter = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")    
    heatmap_globale_a = tf.Variable(tf.zeros((config.map_size, config.map_size), dtype=tf.int32), trainable=False, name="heatmap_globale_a")
    heatmap_globale_b = tf.Variable(tf.zeros((config.map_size, config.map_size), dtype=tf.int32), trainable=False, name="heatmap_globale_b")

    agent = CustomPPO(
        actor_net=actor_net,
        critic_net=value_net,
        lr=config.learning_rate,
        clip_epsilon=config.clip_epsilon,
        entropy_coef=config.entropy_start
    )

    entropy_scheduler = EntropyScheduler(
        agent=agent, 
        start_value=config.entropy_start, 
        end_value=config.entropy_end, 
        decay_steps=config.entropy_decay_steps 
    )

    train_summary_writer = tf.summary.create_file_writer('./logs/train')
    memory = MemoryManager(tf_env, agent)
    
    train_checkpointer = common.Checkpointer(
        ckpt_dir='./checkpoints', max_to_keep=3, actor_network=agent.actor,       
        critic_network=agent.critic, optimizer=agent.optimizer,       
        global_step=train_step_counter, heatmap_globale_a=heatmap_globale_a, heatmap_globale_b=heatmap_globale_b
    )

    train_checkpointer.initialize_or_restore()
    start_step = train_step_counter.numpy()
    print(f"--> Restored state. Global starting step: {start_step}")
    print("=== BEGINNING TRAINING LOOP ===")

    hard_mode_activated = False
    time_step = tf_env.reset()
    
    for i in range(config.num_iterations):
        current_global_step = train_step_counter.numpy()
        print(f"\n--- Iteration {i+1} (Global Step: {current_global_step}) ---")
        
        # --- Hyperparameter and curriculum update ---
        current_entropy = entropy_scheduler.step(current_global_step)

        if current_global_step >= config.curriculum_threshold and not hard_mode_activated:
            print(">>> CURRICULUM LEARNING: Difficulty increased! Unforgiving stag enabled.")
            py_env.set_stag_run_away_after_maul(False)
            hard_mode_activated = True

        # --- A. Data collection ---
        print("A) Data collection (multiple rollout: 4 consecutive episodes)...")
        time_step, fast_heatmap_a, fast_heatmap_b, stats = _collect_rollout(agent, tf_env, memory, time_step, config)

        # --- B/C. PPO training ---
        print("B/C) Estrazione traiettoria e Custom PPO training in progress...")
        trajectories = memory.get_sequential_data()
        heatmap_globale_a.assign_add(fast_heatmap_a)
        heatmap_globale_b.assign_add(fast_heatmap_b)
        
        train_loss, custom_return = _train_ppo_epochs(agent, trajectories, current_global_step, config)
        memory.clear_buffer()

        # --- D. Logging and saving ---
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss/Total_Loss', train_loss, step=current_global_step)
            tf.summary.scalar('Loss/Entropy_Coefficient', current_entropy, step=current_global_step)
            tf.summary.scalar('Metrics/Custom_Average_Return', custom_return, step=current_global_step)
            tf.summary.scalar('Behavior/Stags_Caught', stats['stags'], step=current_global_step)
            tf.summary.scalar('Behavior/Plants_Eaten', stats['plants'], step=current_global_step)
            tf.summary.scalar('Behavior/Maulings_Sustained', stats['mauling'], step=current_global_step)
            
            if current_global_step % config.log_interval == 0:
                print(f"Step {current_global_step} | Current entropy: {current_entropy:.4f}")
                tf.summary.image('Exploration/Heatmap_Agent_A', _generate_annotated_heatmap_tf(heatmap_globale_a.numpy()), step=current_global_step)
                tf.summary.image('Exploration/Heatmap_Agent_B', _generate_annotated_heatmap_tf(heatmap_globale_b.numpy()), step=current_global_step)
        
        train_step_counter.assign_add(1)
        step = train_step_counter.numpy()
        print(f"Global Step: {step} | Custom PPO Loss: {train_loss:.4f}")

        if step > 0 and step % config.save_interval == 0:
            train_checkpointer.save(global_step=step)
            gc.collect()

    print("Training Completed successfully!")


if __name__ == "__main__":
    train_agent()