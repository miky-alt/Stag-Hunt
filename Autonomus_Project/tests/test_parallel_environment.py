import numpy as np
from unittest.mock import Mock, patch

from envs.tf_wrapper import ParallelStagHuntEnv
from training.config import TrainingConfig
from training.train import _activate_curriculum


def test_parallel_worker_exposes_two_agent_vector_specs():
    environment = ParallelStagHuntEnv(seed=123, run_away_after_maul=True)
    try:
        time_step = environment.reset()
        assert time_step.observation.shape == (2, 31)
        assert time_step.reward.shape == (2,)
        assert environment.action_spec().shape == (2,)
        assert environment.observation_spec().shape == (2, 31)

        next_time_step = environment.step(np.zeros(2, dtype=np.int32))
        assert next_time_step.observation.shape == (2, 31)
        assert next_time_step.reward.shape == (2,)
        assert np.asarray(next_time_step.step_type).shape == ()
    finally:
        environment.close()


def test_parallel_environment_count_is_configurable():
    config = TrainingConfig(num_parallel_envs=4)

    assert config.num_parallel_envs == 4


def test_parallel_curriculum_recreates_workers_with_run_away_disabled():
    config = TrainingConfig(num_parallel_envs=2, rollout_steps=8)
    old_environment = Mock()
    old_tf_environment = Mock()
    old_memory = Mock()
    agent = Mock()
    new_environment = Mock()
    new_tf_environment = Mock()
    new_memory = Mock()
    reset_time_step = object()
    new_tf_environment.reset.return_value = reset_time_step

    with patch(
        'training.train._create_training_environment',
        return_value=(new_environment, new_tf_environment, True),
    ) as create_environment, patch(
        'training.train.MemoryManager',
        return_value=new_memory,
    ) as create_memory:
        result = _activate_curriculum(
            config,
            old_environment,
            old_tf_environment,
            old_memory,
            agent,
            is_parallel=True,
        )

    old_environment.close.assert_called_once_with()
    create_environment.assert_called_once_with(
        config,
        run_away_after_maul=False,
    )
    create_memory.assert_called_once_with(
        new_tf_environment,
        agent,
        max_length=9,
    )
    assert result == (
        new_environment,
        new_tf_environment,
        new_memory,
        reset_time_step,
    )


def test_single_environment_curriculum_keeps_existing_environment():
    config = TrainingConfig(num_parallel_envs=1)
    environment = Mock()
    tf_environment = Mock()
    memory = Mock()
    agent = Mock()

    result = _activate_curriculum(
        config,
        environment,
        tf_environment,
        memory,
        agent,
        is_parallel=False,
    )

    environment.set_stag_run_away_after_maul.assert_called_once_with(False)
    assert result == (environment, tf_environment, memory, None)
