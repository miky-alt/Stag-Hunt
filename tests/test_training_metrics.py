from training.config import TrainingConfig
from training.experiment import load_config_file
from training.monitoring.convergence import (
    custom_return_window_is_stable,
    find_convergence_iteration,
    run_convergence_iteration,
)
from training.agent.schedules import annealed_learning_rate
import json
import pytest


def test_config_file_overrides_are_validated(tmp_path):
    """Checks valid config overrides load and unknown fields are rejected."""
    # Arrange valid overrides, then exercise both accepted and rejected inputs.
    config_path = tmp_path / 'config.json'
    config_path.write_text(
        json.dumps({'environment': 'harvest', 'reward_sharing_coef': 0.5}),
        encoding='utf-8',
    )

    assert load_config_file(config_path) == {
        'environment': 'harvest',
        'reward_sharing_coef': 0.5,
    }

    config_path.write_text(json.dumps({'not_a_field': 1}), encoding='utf-8')
    with pytest.raises(ValueError, match='not_a_field'):
        load_config_file(config_path)


def test_annealed_learning_rate_reaches_configured_endpoints():
    """Checks that linear learning-rate annealing reaches both endpoints."""
    # Check the initial, midpoint, and final schedule values for one config.
    config = TrainingConfig(
        num_iterations=11,
        learning_rate=3e-4,
        learning_rate_end=0.0,
    )

    assert annealed_learning_rate(config, 0) == config.learning_rate
    assert annealed_learning_rate(config, 10) == config.learning_rate_end
    assert annealed_learning_rate(config, 5) == config.learning_rate / 2


def test_convergence_requires_custom_return_above_threshold_for_stable_window():
    """Checks that convergence is recorded at the first qualifying return window."""
    # Configure a strict rolling window and include early below-threshold data.
    config = TrainingConfig(
        convergence_custom_return_threshold=300.0,
        convergence_window=3,
        convergence_max_custom_return_std=2.0,
    )
    metrics = [
        {'return': value}
        for value in (290.0, 301.0, 302.0, 300.5, 304.0)
    ]

    assert find_convergence_iteration(metrics, config) == 4


def test_convergence_rejects_unstable_or_subthreshold_custom_return():
    """Checks that noisy or below-threshold returns do not count as convergence."""
    # Compare noisy and subthreshold windows against the same convergence rules.
    config = TrainingConfig(
        convergence_window=3,
        convergence_max_custom_return_std=10.0,
    )
    unstable_metrics = [{'return': value} for value in (301.0, 340.0, 301.0)]
    subthreshold_metrics = [{'return': value} for value in (299.0, 301.0, 302.0)]

    assert find_convergence_iteration(unstable_metrics, config) is None
    assert find_convergence_iteration(subthreshold_metrics, config) is None


def test_current_stability_indicator_can_revert_after_return_drops():
    """Checks that the current stability indicator clears after returns decline."""
    # Establish a stable window, append a low result, and verify the live status.
    config = TrainingConfig(
        convergence_custom_return_threshold=300.0,
        convergence_window=3,
        convergence_max_custom_return_std=2.0,
    )
    metrics = [{'return': value} for value in (301.0, 302.0, 301.0)]

    assert custom_return_window_is_stable(metrics, config) is True
    metrics.append({'return': 250.0})
    assert custom_return_window_is_stable(metrics, config) is False


def test_run_convergence_iteration_reads_current_and_legacy_result_keys():
    """Checks that convergence metadata supports current and legacy result keys."""
    # Prefer the current result key while retaining legacy and missing-key behavior.
    assert run_convergence_iteration({'first_stable_window_iteration': 25}) == 25
    assert run_convergence_iteration({'convergence_iteration': 30}) == 30
    assert run_convergence_iteration({'first_stable_window_iteration': None}) is None
