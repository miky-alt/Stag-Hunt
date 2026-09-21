import os
# Force compatibility with Keras 2 to avoid internal TensorFlow errors
os.environ["TF_USE_LEGACY_KERAS"] = "1" 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import gc
import time
import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from tf_agents.environments import wrappers, tf_py_environment
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

# Import our custom modules
from envs.tf_wrapper import TFStagHuntWrapper
from agents.networks import ActorNetwork, CriticNetwork
from agents.ppo_custom import CustomPPO
from training.memory import MemoryManager
from tf_agents.trajectories.policy_step import PolicyStep

import csv

class ScatolaNeraLogger:
    """
    Gestisce il logging locale degli episodi salvando i dati fisici della griglia.
    """
    def __init__(self, log_dir="./logs_csv", map_size=(5, 5)):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_corrente = None
        self.writer = None
        self.map_size = map_size
        self.reset_episode_counters()

    def reset_episode_counters(self):
        self.heatmap_a = np.zeros(self.map_size, dtype=int)
        self.heatmap_b = np.zeros(self.map_size, dtype=int)
        self.conteggio_cervo = 0
        self.conteggio_pianta = 0
        self.conteggio_mauling = 0
        self.snapshot_eventi = []

    def start_episode(self, episode_number):
        self.reset_episode_counters()
        path = os.path.join(self.log_dir, f"episodio_{numero_episodio}.csv")
        self.file_corrente = open(path, mode='w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file_corrente)
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
            self.conteggio_cervo += 1
            evento_str = "CERVO CATTURATO"
            self.snapshot_eventi.append(f"Stag at Step {step_idx} | Pos A:{pos_a}, Pos B:{pos_b}, Stag:{positions.get('Stag')}")
        elif 1.0 <= reward <= 2.0:
            self.conteggio_pianta += 1
            evento_str = "PIANTA"
        elif reward < 0.0:
            self.conteggio_mauling += 1
            evento_str = "MAULING (Cornata)"
            self.snapshot_eventi.append(f"Mauling at Step {step_idx} | Pos A:{pos_a}, Pos B:{pos_b}, Stag:{positions.get('Stag')}")

        self.writer.writerow([
            step_idx, actions[0], actions[1], pos_a, pos_b,
            positions.get('Stag', 'N/D'), positions.get('Plant', 'N/D'),
            probs_a, probs_b, reward, evento_str
        ])

    def close_episode(self):
        if self.file_corrente:
            self.writer.writerow([])
            self.writer.writerow(["--- RIASSUNTO COMPORTAMENTALE ---"])
            self.writer.writerow(["Totale Cervi Catturati:", self.conteggio_cervo])
            self.writer.writerow(["Totale Piante Mangiate:", self.conteggio_pianta])
            self.writer.writerow(["Totale Mauling Subiti:", self.conteggio_mauling])
            self.writer.writerow([])
            self.writer.writerow(["--- SNAPSHOT EVENTI CRITICI ---"])
            for snap in self.snapshot_eventi: self.writer.writerow([snap])
            self.writer.writerow([])
            self.writer.writerow(["--- HEATMAP ESPLORAZIONE AGENTE A ---"])
            for riga in self.heatmap_a: self.writer.writerow(riga.tolist())
            self.writer.writerow([])
            self.writer.writerow(["--- HEATMAP ESPLORAZIONE AGENTE B ---"])
            for riga in self.heatmap_b: self.writer.writerow(riga.tolist())

            self.file_corrente.close()
            self.file_corrente = None


class EntropyScheduler:
    def __init__(self, agent, start_value=0.15, end_value=0.001, decay_steps=10000):
        self.agent = agent
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self, current_step):
        if current_step >= self.decay_steps:
            nuova_entropia = self.end_value
        else:
            frazione_completata = current_step / self.decay_steps
            nuova_entropia = self.start_value - frazione_completata * (self.start_value - self.end_value)
            
        self.agent.entropy_coef.assign(nuova_entropia)
        return nuova_entropia


def _generate_annotated_heatmap_tf(heatmap_counts):
    somma = np.sum(heatmap_counts)
    griglia_perc = (heatmap_counts / somma) * 100 if somma > 0 else heatmap_counts
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(griglia_perc, cmap='Blues', vmin=0, vmax=100)
    
    for r in range(griglia_perc.shape[0]):
        for c in range(griglia_perc.shape[1]):
            val = griglia_perc[r, c]
            colore_testo = "white" if val > 45.0 else "black"
            ax.text(c, r, f"{val:.1f}%", va='center', ha='center', color=colore_testo, fontsize=9, weight='bold')
    
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    immagine_tensor = tf.image.decode_png(buf.getvalue(), channels=4)
    return tf.expand_dims(immagine_tensor, 0)


def train_agent():
    print("1. Initializing Environment...")
    logger_csv = ScatolaNeraLogger()

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
    train_step_counter = tf.Variable(0, dtype=tf.int64)

    # --- GLOBAL VARIABLES FOR HISTORICAL HEATMAPS ---
    heatmap_globale_a = tf.Variable(tf.zeros((5, 5), dtype=tf.int32), trainable=False, name="heatmap_globale_a")
    heatmap_globale_b = tf.Variable(tf.zeros((5, 5), dtype=tf.int32), trainable=False, name="heatmap_globale_b")

    agent = CustomPPO(
        actor_net=actor_net,
        critic_net=value_net,
        lr=3e-4,
        clip_epsilon=0.2,
        entropy_coef=0.15 # Start with high exploration.
    )

    # Decay scaled mathematically for four episodes per iteration.
    schedulatore_entropia = EntropyScheduler(
        agent=agent, start_value=0.15, end_value=0.01, decay_steps=10000 
    )

    summary_dir = './logs/train'
    train_summary_writer = tf.summary.create_file_writer(summary_dir)
    memory = MemoryManager(tf_env, agent)
    
    train_checkpointer = common.Checkpointer(
        ckpt_dir='./checkpoints', max_to_keep=3, actor_network=agent.actor,       
        critic_network=agent.critic, optimizer=agent.optimizer,       
        global_step=train_step_counter, heatmap_globale_a=heatmap_globale_a, heatmap_globale_b=heatmap_globale_b
    )

    train_checkpointer.initialize_or_restore()
    start_step = train_step_counter.numpy()
    print(f"--> Stato ricaricato. Punto di partenza globale: Step {start_step}")

    print("=== BEGINNING TRAINING LOOP ===")
    num_iterations = 16000  # Compensate for the increased rollouts (16k * 260 = 4.1M steps).
    hard_mode_activated = False
    time_step = tf_env.reset()
    
    for i in range(num_iterations):
        current_global_step = train_step_counter.numpy()
        print(f"\n--- Inizio Iterazione {i+1} (Global Step: {current_global_step}) ---")
        
        entropia_attuale = schedulatore_entropia.step(current_global_step)

        if current_global_step % 200 == 0:
            print(f"Step {current_global_step} | Entropia attuale: {entropia_attuale:.4f}")

        # Curriculum learning: mathematically, 15000 / 4 = 3750.
        if current_global_step >= 3750 and not hard_mode_activated:
            print(">>> CURRICULUM LEARNING: Aumento difficoltà! Stag implacabile attivato.")
            py_env.set_stag_run_away_after_maul(False)
            hard_mode_activated = True
        
        heatmap_veloce_a = np.zeros((5, 5), dtype=int)
        heatmap_veloce_b = np.zeros((5, 5), dtype=int)
        cervi_it, piante_it, mauling_it = 0, 0, 0

        print("A) Raccolta Dati (Rollout Multiplo: 4 Episodi di fila)...")
        # 260 steps = four consecutive games before updating weights (4x speed).
        for step_idx in range(260):
            action_distribution, _ = agent.actor(time_step.observation)
            raw_action = action_distribution.sample()
            action_tensor = tf.cast(raw_action, tf.int32)
            action_step = PolicyStep(action=action_tensor)
            
            next_time_step = tf_env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            memory.replay_buffer.add_batch(traj)
            
            mappa_piatta = next_time_step.observation.numpy()[0]
            xa, ya = min(int(mappa_piatta[0]), 4), min(int(mappa_piatta[1]), 4)
            xb, yb = min(int(mappa_piatta[2]), 4), min(int(mappa_piatta[3]), 4)
            
            heatmap_veloce_a[xa, ya] += 1
            heatmap_veloce_b[xb, yb] += 1

            r_passato = float(tf.reduce_sum(time_step.reward))
            if r_passato >= 10.0: cervi_it += 1
            elif 1.0 <= r_passato <= 2.0: piante_it += 1
            elif r_passato < 0.0: mauling_it += 1

            time_step = next_time_step

        print("B) Estrazione traiettoria cronologica dal Buffer...")
        trajectories = memory.get_sequential_data()

        print("C) Custom PPO training in progress...")
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

        # Global accumulation.
        heatmap_globale_a.assign_add(heatmap_veloce_a)
        heatmap_globale_b.assign_add(heatmap_veloce_b)
        
        # Reduce chart updates for performance.
        if current_global_step % 200 == 0:
            tensor_hd_a = _generate_annotated_heatmap_tf(heatmap_globale_a.numpy())
            tensor_hd_b = _generate_annotated_heatmap_tf(heatmap_globale_b.numpy())
            with train_summary_writer.as_default():
                tf.summary.image('Exploration/Heatmap_Agent_A', tensor_hd_a, step=current_global_step)
                tf.summary.image('Exploration/Heatmap_Agent_B', tensor_hd_b, step=current_global_step)
        
        # Backpropagation
        with train_summary_writer.as_default():
            ppo_epochs = 4
            for epoch in range(ppo_epochs):
                train_loss = agent.train_step(
                    states=s_t, actions=a_t, old_log_probs=old_log_probs, 
                    rewards=r_t, next_states=s_t_next, dones=dones,
                    global_step=tf.cast(current_global_step, tf.int64)
                )
        
        memory.clear_buffer()

        with train_summary_writer.as_default():
            tf.summary.scalar('Loss/Total_Loss', train_loss, step=current_global_step)
            tf.summary.scalar('Loss/Entropy_Coefficient', entropia_attuale, step=current_global_step)
            
            custom_return = tf.reduce_mean(tf.reduce_sum(r_t, axis=1))
            tf.summary.scalar('Metrics/Custom_Average_Return', custom_return, step=current_global_step)
            tf.summary.scalar('Behavior/Stags_Caught', cervi_it, step=current_global_step)
            tf.summary.scalar('Behavior/Plants_Eaten', piante_it, step=current_global_step)
            tf.summary.scalar('Behavior/Maulings_Sustained', mauling_it, step=current_global_step)
        
        train_step_counter.assign_add(1)
        step = train_step_counter.numpy()
        print(f"Global Step: {step} | Custom PPO Loss: {train_loss:.4f}")

        if step > 0 and step % 100 == 0:
            train_checkpointer.save(global_step=step)
            gc.collect()

    print("Training Completed successfully!")


if __name__ == "__main__":
    train_agent()