import numpy as np
from unittest.mock import Mock, patch

from envs.tf_wrapper import ParallelStagHuntEnv, TFStagHuntWrapper
from training.config import TrainingConfig
from training.environment.curriculum import (
    activate_curriculum,
    stag_follow_probability,
)
from training.environment.factory import create_training_environment


def test_parallel_worker_exposes_two_agent_vector_specs():
    """Checks that one worker exposes vector observations, actions, and rewards."""
    # Create a worker, inspect its initial batch, and take one joint action.
    environment = ParallelStagHuntEnv(
        seed=123,
        run_away_after_maul=True,
        handcrafted_features=True,
    )
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
    """Checks that the training configuration accepts multiple workers."""
    # Configure multiple workers and verify the requested count is retained.
    config = TrainingConfig(num_parallel_envs=4)

    assert config.num_parallel_envs == 4


def test_stag_follow_curriculum_probability_ramps_then_returns_to_default():
    """Checks that stag-follow probability ramps to one and then resets to default."""
    # Check the start, midpoint, endpoint, and post-curriculum schedule states.
    config = TrainingConfig(
        stag_follow_curriculum_enabled=True,
        stag_follow_curriculum_steps=5,
    )

    assert stag_follow_probability(config, 0) == 0.0
    assert stag_follow_probability(config, 2) == 0.5
    assert stag_follow_probability(config, 4) == 1.0
    assert stag_follow_probability(config, 5) is None


def test_stag_follow_curriculum_step_mode_keeps_stag_stationary_then_restores_default():
    """Checks that step mode holds probability at zero and then resets to default."""
    config = TrainingConfig(
        stag_follow_curriculum_enabled=True,
        stag_follow_curriculum_steps=5,
        stag_follow_curriculum_mode='step',
    )

    assert stag_follow_probability(config, 0) == 0.0
    assert stag_follow_probability(config, 4) == 0.0
    assert stag_follow_probability(config, 5) is None


def test_stag_follow_curriculum_is_disabled_by_default_or_master_flag():
    """Checks that stag-follow curriculum requires both enable flags."""
    # Verify both the default configuration and master switch suppress scheduling.
    assert stag_follow_probability(TrainingConfig(), 0) is None
    config = TrainingConfig(
        curriculum_enabled=False,
        stag_follow_curriculum_enabled=True,
    )
    assert stag_follow_probability(config, 0) is None


def test_stag_is_stationary_at_zero_follow_probability_and_follows_at_one():
    """Checks that zero probability stops pursuit and full probability enables it."""
    # Use deterministic randomness to compare disabled, enabled, and default behavior.
    environment = TFStagHuntWrapper(
        env_name='StagHunt-Hunt-v0',
        seed=123,
        stag_follow_probability=0.0,
    )
    game = environment.env.unwrapped.game
    stag_position = game.STAG.copy()
    try:
        with patch('gymnasium_stag_hunt.src.games.staghunt_game.random', return_value=0.5):
            with patch.object(game, '_seek_agent') as seek_agent:
                game._move_stag()
        seek_agent.assert_not_called()
        assert np.array_equal(game.STAG, stag_position)

        environment.set_stag_follow_probability(1.0)
        with patch('gymnasium_stag_hunt.src.games.staghunt_game.random', return_value=0.5):
            with patch.object(game, '_seek_agent', return_value=stag_position) as seek_agent:
                game._move_stag()
        seek_agent.assert_called_once()

        environment.set_stag_follow_probability(None)
        with patch.object(game, '_seek_agent', return_value=stag_position) as seek_agent:
            game._move_stag()
        seek_agent.assert_called_once()
    finally:
        environment.close()


def test_harvest_and_escalation_handcrafted_radars():
    """Checks handcrafted radar feature sizes and values in both environments."""
    # Validate Harvest's plant/partner radar, then verify Escalation's mark radar.
    harvest = TFStagHuntWrapper(env_name='StagHunt-Harvest-v0', seed=1, handcrafted_features=True)
    try:
        assert harvest.observation_spec().shape == (49,)
        assert harvest.reset().observation.shape == (2, 49)
        # me (0,0), partner (4,4); young plant (1,0), mature plant (0,3), two young plants on (0,0).
        obs = np.array([0, 0, 4, 4, 1, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        radar = harvest._inject_harvest_radar(obs)[16:]
        assert radar[:3].tolist() == [8.0, 4.0, 4.0]
        assert radar[3:9].tolist() == [1.0, 1.0, 0.0, 7.0, -3.0, -4.0]
        assert radar[-6:-3].tolist() == [3.0, 0.0, 3.0]  # nearest mature
        assert radar[-3:].tolist() == [0.0, 0.0, 0.0]  # nearest young
    finally:
        harvest.close()

    escalation = TFStagHuntWrapper(env_name='StagHunt-Escalation-v0', seed=1, handcrafted_features=True)
    try:
        assert escalation.observation_spec().shape == (15,)
        radar = escalation._inject_escalation_radar(np.array([0, 0, 2, 2, 1, 3], dtype=np.float32))[6:]
        assert radar.tolist() == [4.0, 2.0, 2.0, 4.0, 1.0, 3.0, 2.0, -1.0, 1.0]
    finally:
        escalation.close()


def test_harvest_wrapper_reports_young_and_mature_plant_events():
    """Reports distinct young and mature plants harvested in the same step."""
    # Keep maturity changes disabled and assign each agent a different plant.
    harvest = TFStagHuntWrapper(
        env_name='StagHunt-Harvest-v0',
        seed=1,
        chance_to_mature=0.0,
        chance_to_die=0.0,
    )
    try:
        harvest.reset()
        game = harvest.env.unwrapped.game
        game.A_AGENT = [1, 1]
        game.B_AGENT = [3, 3]
        game._plants[0] = [1, 1]
        game._plants[1] = [3, 3]
        game._maturity_flags = [True, False, False, False]

        harvest.step(np.array([4, 4], dtype=np.int32))

        assert harvest.last_info == {
            'plants_harvested_young': 1,
            'plants_harvested_mature': 1,
            'plants_harvested_young_by_agent': [0, 1],
            'plants_harvested_mature_by_agent': [1, 0],
        }
    finally:
        harvest.close()


def test_stag_following_can_be_disabled_for_training_environment():
    """Checks that environment creation forwards the configured stag-following flag."""
    # Mock construction and inspect the keyword arguments sent to the wrapper.
    config = TrainingConfig(stag_follows=False, num_parallel_envs=1)

    with patch('training.environment.factory.TFStagHuntWrapper') as wrapper, patch(
        'training.environment.factory.tf_py_environment.TFPyEnvironment'
    ):
        wrapper.return_value = Mock()
        create_training_environment(config)

    wrapper.assert_called_once_with(
        env_name='StagHunt-Hunt-v0',
        seed=42,
        stag_follows=False,
        run_away_after_maul=False,
        handcrafted_features=False,
    )


def test_parallel_curriculum_recreates_workers_with_run_away_disabled():
    """Checks that parallel curriculum recreates workers and resets rollout memory."""
    # Replace factory and buffer creation, then verify curriculum handoff state.
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
        'training.environment.factory.create_training_environment',
        return_value=(new_environment, new_tf_environment, True),
    ) as create_environment, patch(
        'training.environment.curriculum.MemoryManager',
        return_value=new_memory,
    ) as create_memory:
        result = activate_curriculum(
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
    """Checks that single-environment curriculum updates the existing Hunt wrapper."""
    # Activate the phase on the existing wrapper and check the returned reset step.
    config = TrainingConfig(num_parallel_envs=1)
    environment = Mock()
    tf_environment = Mock()
    reset_time_step = object()
    tf_environment.reset.return_value = reset_time_step
    memory = Mock()
    agent = Mock()

    result = activate_curriculum(
        config,
        environment,
        tf_environment,
        memory,
        agent,
        is_parallel=False,
    )

    environment.set_stag_run_away_after_maul.assert_called_once_with(False)
    tf_environment.reset.assert_called_once_with()
    assert result == (environment, tf_environment, memory, reset_time_step)
