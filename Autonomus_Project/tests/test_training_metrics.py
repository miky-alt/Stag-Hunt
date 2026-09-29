from training.config import TrainingConfig
from training.train import _annealed_learning_rate, _find_convergence_iteration


def test_annealed_learning_rate_reaches_configured_endpoints():
    config = TrainingConfig(
        num_iterations=11,
        learning_rate=3e-4,
        learning_rate_end=0.0,
    )

    assert _annealed_learning_rate(config, 0) == config.learning_rate
    assert _annealed_learning_rate(config, 10) == config.learning_rate_end
    assert _annealed_learning_rate(config, 5) == config.learning_rate / 2


def test_convergence_is_detected_per_agent_after_stable_window():
    config = TrainingConfig(
        rollout_steps=11,
        convergence_reward_rate=0.4,
        convergence_window=3,
        convergence_max_rate_std=0.04,
    )
    metrics = [
        {'agent_a_return': value, 'agent_b_return': value}
        for value in (2.0, 4.0, 4.2, 4.1, 4.0)
    ]

    assert _find_convergence_iteration(metrics, config, 'agent_a_return') == 4
    assert _find_convergence_iteration(metrics, config, 'agent_b_return') == 4


def test_convergence_rejects_unstable_reward_window():
    config = TrainingConfig(
        rollout_steps=11,
        convergence_reward_rate=0.4,
        convergence_window=3,
        convergence_max_rate_std=0.04,
    )
    metrics = [
        {'agent_a_return': value}
        for value in (4.0, 8.0, 4.0, 8.0, 4.0)
    ]

    assert _find_convergence_iteration(metrics, config, 'agent_a_return') is None
