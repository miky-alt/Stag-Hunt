from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and experiment configuration."""
    seed: int = 42
    environment: str = 'hunt'
    num_iterations: int = 16_000
    rollout_steps: int = 260
    num_parallel_envs: int = 1
    agent_architecture: str = 'shared_shared'
    critic_observation: str = 'local'
    reward_sharing_coef: float = 0.0
    minibatch_size: int = 65
    ppo_epochs: int = 4
    curriculum_threshold: int = 3750
    log_interval: int = 200
    save_interval: int = 100
    map_size: int = 5
    learning_rate: float = 3e-4
    learning_rate_end: float = 0.0
    clip_epsilon: float = 0.2
    value_loss_coef: float = 1.0
    value_normalization: bool = False
    value_clipping: bool = False
    value_clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_start: float = 0.15
    entropy_end: float = 0.01
    entropy_decay_steps: int = 10000
    convergence_reward_rate: float = 0.4
    convergence_window: int = 5
    convergence_max_rate_std: float = 0.04
