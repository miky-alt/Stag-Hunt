"""Construction of the single or parallel TF-Agents training environment."""
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.system import multiprocessing as tf_multiprocessing

from envs.registry import resolve_environment_id
from envs.tf_wrapper import ParallelStagHuntEnv, TFStagHuntWrapper
from training.config import TrainingConfig

_PARALLEL_MP_INITIALIZED = False


def environment_kwargs_for(config: TrainingConfig, run_away_after_maul=None):
    """Assembles the wrapper options relevant to the configured game."""
    if run_away_after_maul is None:
        run_away_after_maul = config.run_away_after_maul
    kwargs = {'handcrafted_features': config.handcrafted_features}
    if config.environment == 'hunt':
        kwargs['stag_follows'] = config.stag_follows
        kwargs['run_away_after_maul'] = run_away_after_maul
        if config.curriculum_enabled and config.stag_follow_curriculum_enabled:
            kwargs['stag_follow_probability'] = 0.0
    return kwargs


def create_training_environment(config: TrainingConfig, run_away_after_maul=None):
    """Creates the configured environment; returns (py_env, tf_env, is_parallel)."""
    global _PARALLEL_MP_INITIALIZED
    env_name = resolve_environment_id(config.environment)
    environment_kwargs = environment_kwargs_for(config, run_away_after_maul)

    # Use a direct wrapper for one game; otherwise initialize seeded workers.
    if config.num_parallel_envs <= 1:
        py_env = TFStagHuntWrapper(
            env_name=env_name,
            seed=config.seed,
            **environment_kwargs,
        )
        return py_env, tf_py_environment.TFPyEnvironment(py_env), False

    if not _PARALLEL_MP_INITIALIZED:
        tf_multiprocessing.enable_interactive_mode()
        _PARALLEL_MP_INITIALIZED = True
    # Give each worker a deterministic but distinct initial seed.
    seeds = [config.seed + environment_index for environment_index in range(config.num_parallel_envs)]
    env_constructors = [
        lambda seed=seed: ParallelStagHuntEnv(
            env_name=env_name,
            seed=seed,
            **environment_kwargs,
        )
        for seed in seeds
    ]
    py_env = parallel_py_environment.ParallelPyEnvironment(
        env_constructors,
        start_serially=True,
        blocking=False,
        flatten=True,
    )
    return py_env, tf_py_environment.TFPyEnvironment(py_env), True
