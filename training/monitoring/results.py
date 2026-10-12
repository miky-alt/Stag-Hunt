"""Builds the training_results.json payloads for one run and for a multi-seed group."""
import numpy as np
import tensorflow as tf

from training.monitoring.convergence import (
    custom_return_window_is_stable,
    find_convergence_iteration,
    run_convergence_iteration,
)

AGGREGATED_METRICS = (
    'return', 'agent_a_return', 'agent_b_return', 'loss', 'entropy', 'stags_caught',
    'plants_eaten', 'maulings_sustained', 'learning_rate', 'approx_kl',
    'clip_fraction', 'policy_loss', 'value_loss', 'policy_entropy', 'gradient_norm',
    'stag_follow_probability',
)


def iteration_metrics(custom_return, agent_returns, train_loss, entropy_coefficient,
                      stag_follow_probability, stats, diagnostics):
    """Flattens one iteration's tensors and counters into a JSON-friendly dict."""
    return {
        'return': float(custom_return.numpy()),
        'agent_a_return': float(tf.reduce_mean(agent_returns[:, 0]).numpy()),
        'agent_b_return': float(tf.reduce_mean(agent_returns[:, 1]).numpy()),
        'loss': float(train_loss.numpy() if hasattr(train_loss, 'numpy') else train_loss),
        'entropy': float(entropy_coefficient),
        'stag_follow_probability': float(stag_follow_probability),
        'stags_caught': float(stats['stags']),
        'plants_eaten': float(stats['plants']),
        'maulings_sustained': float(stats['mauling']),
        'learning_rate': float(diagnostics['learning_rate'].numpy()),
        'approx_kl': float(diagnostics['approx_kl'].numpy()),
        'clip_fraction': float(diagnostics['clip_fraction'].numpy()),
        'policy_loss': float(diagnostics['policy_loss'].numpy()),
        'value_loss': float(diagnostics['value_loss'].numpy()),
        'policy_entropy': float(diagnostics['entropy'].numpy()),
        'gradient_norm': float(diagnostics['gradient_norm'].numpy()),
    }


def run_results(experiment_id, config, training_metrics, heatmap_a, heatmap_b):
    """Per-run summary: behaviour totals, convergence and cumulative heatmaps."""
    return {
        'experiment_id': experiment_id,
        'seed': config.seed,
        'iterations': len(training_metrics),
        'total_stags_caught': sum(m['stags_caught'] for m in training_metrics),
        'total_plants_eaten': sum(m['plants_eaten'] for m in training_metrics),
        'total_maulings_sustained': sum(m['maulings_sustained'] for m in training_metrics),
        'first_stable_window_iteration': find_convergence_iteration(training_metrics, config),
        'stable_window_at_end': bool(custom_return_window_is_stable(training_metrics, config)),
        'convergence_definition': {
            'metric': 'Custom_Average_Return',
            'return_threshold': config.convergence_custom_return_threshold,
            'stable_window': config.convergence_window,
            'max_return_std': config.convergence_max_custom_return_std,
            'interpretation': 'rolling stability indicator; may return to false',
        },
        'heatmap_agent_a': np.asarray(heatmap_a).tolist(),
        'heatmap_agent_b': np.asarray(heatmap_b).tolist(),
    }


def group_results(group_id, config, seeds, training_runs, run_totals):
    """Aggregates per-seed curves and totals: mean/std by iteration and final values."""
    convergence_iterations = [
        iteration for iteration in (run_convergence_iteration(run) for run in run_totals)
        if iteration is not None
    ]
    heatmap_a = sum(np.asarray(run['heatmap_agent_a'], dtype=int) for run in run_totals)
    heatmap_b = sum(np.asarray(run['heatmap_agent_b'], dtype=int) for run in run_totals)

    aggregate_metrics = {}
    for metric in AGGREGATED_METRICS:
        curve = np.asarray([[iteration[metric] for iteration in run] for run in training_runs], dtype=float)
        final_values = np.asarray([run[-1][metric] if run else float('nan') for run in training_runs], dtype=float)
        aggregate_metrics[metric] = {
            'final_mean': float(np.nanmean(final_values)),
            'final_std': float(np.nanstd(final_values, ddof=1)) if len(final_values) > 1 else 0.0,
            'mean_by_iteration': np.nanmean(curve, axis=0).tolist(),
            'std_by_iteration': (
                np.nanstd(curve, axis=0, ddof=1) if len(training_runs) > 1 else np.zeros(curve.shape[1])
            ).tolist(),
            'final_by_seed': {str(seed): float(value) for seed, value in zip(seeds, final_values)},
        }

    return {
        'experiment_id': group_id,
        'seeds': seeds,
        'iterations': len(training_runs[0]) if training_runs else 0,
        'metrics': aggregate_metrics,
        'total_stags_caught': sum(run['total_stags_caught'] for run in run_totals),
        'total_plants_eaten': sum(run['total_plants_eaten'] for run in run_totals),
        'total_maulings_sustained': sum(run['total_maulings_sustained'] for run in run_totals),
        'first_stable_window_iteration_mean': (
            float(np.mean(convergence_iterations)) if convergence_iterations else None
        ),
        'first_stable_window_iteration_std': (
            float(np.std(convergence_iterations, ddof=1)) if len(convergence_iterations) > 1 else 0.0
        ),
        'stable_seeds': len(convergence_iterations),
        'convergence_definition': {
            'metric': 'Custom_Average_Return',
            'return_threshold': config.convergence_custom_return_threshold,
            'stable_window': config.convergence_window,
            'max_return_std': config.convergence_max_custom_return_std,
        },
        'heatmap_agent_a': heatmap_a.tolist() if run_totals else [],
        'heatmap_agent_b': heatmap_b.tolist() if run_totals else [],
    }
