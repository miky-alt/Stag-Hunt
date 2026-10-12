"""Rolling stability criterion on the training return."""
import numpy as np


def custom_return_window_is_stable(training_metrics, config):
    """True if the last `convergence_window` returns are all above threshold and low-variance."""
    window_size = config.convergence_window
    if len(training_metrics) < window_size:
        return None
    returns = np.asarray([metric['return'] for metric in training_metrics[-window_size:]], dtype=float)
    return bool(
        np.all(returns >= config.convergence_custom_return_threshold)
        and np.std(returns) <= config.convergence_max_custom_return_std
    )


def find_convergence_iteration(training_metrics, config):
    """First 1-based iteration whose full window meets the criterion, or None."""
    window_size = config.convergence_window
    if len(training_metrics) < window_size:
        return None
    for end_index in range(window_size, len(training_metrics) + 1):
        if custom_return_window_is_stable(training_metrics[:end_index], config):
            return end_index
    return None


def run_convergence_iteration(run_results):
    """Reads the convergence iteration from current or legacy per-run results."""
    return run_results.get('first_stable_window_iteration', run_results.get('convergence_iteration'))
