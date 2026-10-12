"""Hunt curricula: the unforgiving-stag phase switch and the stag-mobility schedule."""
from training.config import TrainingConfig
from training.environment import factory
from training.rollout.memory import MemoryManager


def set_stag_follow_probability(py_env, is_parallel, probability):
    """Updates the stag-follow probability in one environment or in all workers."""
    if is_parallel:
        # Dispatch the mutation to every worker and wait for all updates to finish.
        promises = [
            worker.call('set_stag_follow_probability', probability)
            for worker in py_env.envs
        ]
        for promise in promises:
            promise()
    else:
        py_env.set_stag_follow_probability(probability)


def stag_follow_probability(config: TrainingConfig, iteration):
    """Returns the scheduled follow probability, or None for the game default."""
    if (
        config.environment != 'hunt'
        or not config.curriculum_enabled
        or not config.stag_follow_curriculum_enabled
    ):
        return None
    if config.stag_follow_curriculum_steps < 1:
        raise ValueError('stag_follow_curriculum_steps must be at least 1.')
    if iteration >= config.stag_follow_curriculum_steps:
        return None
    if config.stag_follow_curriculum_mode == 'step':
        return 0.0
    if config.stag_follow_curriculum_mode != 'linear':
        raise ValueError(
            f"stag_follow_curriculum_mode must be 'linear' or 'step', got {config.stag_follow_curriculum_mode!r}."
        )
    if config.stag_follow_curriculum_steps == 1:
        return 1.0
    return iteration / (config.stag_follow_curriculum_steps - 1)


def activate_curriculum(config: TrainingConfig, py_env, tf_env, memory, agent, is_parallel):
    """Switches Hunt to its curriculum phase; returns (py_env, tf_env, memory, reset_time_step)."""
    if config.environment != 'hunt':
        return py_env, tf_env, memory, None
    if is_parallel:
        # Recreate parallel workers and their buffer under the updated rule.
        py_env.close()
        py_env, tf_env, _ = factory.create_training_environment(
            config,
            run_away_after_maul=False,
        )
        memory = MemoryManager(
            tf_env,
            agent,
            max_length=config.rollout_steps + 1,
        )
        return py_env, tf_env, memory, tf_env.reset()

    # Update the existing single wrapper and start the new phase at episode reset.
    py_env.set_stag_run_away_after_maul(False)
    return py_env, tf_env, memory, tf_env.reset()
