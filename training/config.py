from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters and experiment configuration."""
    seed: int = 42
    environment: str = 'hunt'
    handcrafted_features: bool = False
    stag_follows: bool = True
    run_away_after_maul: bool = False
    stag_follow_curriculum_enabled: bool = False
    stag_follow_curriculum_steps: int = 1000
    # 'linear': follow probability ramps 0->1; 'step': stag stationary, then default pursuit.
    stag_follow_curriculum_mode: str = 'linear'
    num_iterations: int = 16_000
    rollout_steps: int = 256
    num_parallel_envs: int = 1
    agent_architecture: str = 'shared_shared'
    network_init: str = 'orthogonal'
    critic_observation: str = 'local'
    reward_sharing_coef: float = 0.0
    minibatch_size: int = 64
    ppo_epochs: int = 4
    curriculum_threshold: int = 4_000
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
    curriculum_enabled: bool = True
    convergence_custom_return_threshold: float = 300.0
    convergence_window: int = 5
    convergence_max_custom_return_std: float = 10.0
