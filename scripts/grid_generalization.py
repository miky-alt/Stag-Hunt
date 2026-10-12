"""Extra evaluations of the final policies: other grid sizes and/or greedy actions.

Usage (from the project root):
    python scripts\\grid_generalization.py hunt --grid-size 7            # 7x7, stochastic
    python scripts\\grid_generalization.py hunt --deterministic          # 5x5, greedy (argmax)
    python scripts\\grid_generalization.py hunt --grid-size 7 --deterministic
    python scripts\\grid_generalization.py hunt --grid-size 7 --skip-evaluation   # redraw only

Every setting is compared against the main study evaluation (5x5, stochastic).
Policies were trained on the 5x5 grid, so another grid is an out-of-distribution
test. Results are written next to the default evaluation under a setting-specific
name, so the main study files are never overwritten.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import report_plots as rp  # noqa: E402
from final_ablations import DEFAULT_EVAL_SEEDS, run_logged  # noqa: E402

TRAINING_GRID = 5


def _setting(grid_size, deterministic):
    """Filename suffix and human label for one evaluation setting."""
    parts = []
    if grid_size != TRAINING_GRID:
        parts.append(f'grid{grid_size}')
    if deterministic:
        parts.append('greedy')
    suffix = '_'.join(parts)
    label = f'{grid_size}x{grid_size}, ' + ('greedy' if deterministic else 'stochastic')
    return suffix, label


def _evaluate_missing(environment, output_root, grid_size, deterministic, args):
    """Run the evaluation setting for every seed run that does not have it yet."""
    log_dir = output_root / 'logs'
    suffix, _ = _setting(grid_size, deterministic)
    filename = f'evaluation_results_{suffix}.json'
    for group_dir in sorted(output_root.glob(f'final_{environment}_*')):
        if not (group_dir / 'training_results.json').exists():
            continue
        for run_dir in sorted((group_dir / 'runs').glob('seed_*')):
            if (run_dir / filename).exists():
                print(f'[skip] {run_dir.relative_to(output_root).as_posix()} already has {filename}')
                continue
            run_name = run_dir.relative_to(output_root).as_posix()
            print(f'[eval {suffix}] {run_name}')
            command = [
                sys.executable, '-m', 'evaluate.evaluate',
                '--experiments-root', str(output_root),
                '--experiment', run_name,
                '--seeds', args.eval_seeds,
                '--episodes-per-seed', str(args.episodes_per_seed),
                '--output-file', filename,
            ]
            if grid_size != TRAINING_GRID:
                command += ['--grid-size', str(grid_size)]
            if deterministic:
                command.append('--deterministic')
            return_code = run_logged(
                command,
                log_dir / f'eval_{suffix}_{run_name.replace("/", "_")}.log',
            )
            if return_code != 0:
                raise RuntimeError(f'Evaluation failed: {run_name}')


def _seed_means(group, filename):
    """Per-training-seed mean evaluation reward read from the given result file."""
    values = []
    for run in group['runs']:
        if run['evaluation'] is None:
            continue
        path = run['path'] / filename
        if path.exists():
            values.append(rp._read_json(path)['mean_reward'])
    return values


def _load_groups_with_paths(root, environment):
    """report_plots.load_groups plus the run directory, needed to find extra result files."""
    groups = rp.load_groups(root, environment)
    for group in groups:
        group_dir = next(Path(root).glob(group['experiment_id']))
        for run in group['runs']:
            run['path'] = group_dir / 'runs' / f"seed_{run['seed']}"
    return groups


def plot_comparison(groups, colors, filename, setting_label, random_reference=None):
    """Grouped bars: main study evaluation (5x5, stochastic) vs the given setting."""
    labels = [group['label'] for group in groups]
    base = [rp._mean_std(rp.evaluation_rewards_by_training_seed(group)) for group in groups]
    other = [rp._mean_std(_seed_means(group, filename)) for group in groups]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(groups)), 4.0))
    for index, group in enumerate(groups):
        color = colors[group['label']]
        ax.bar(x[index] - width / 2, base[index][0], width, yerr=base[index][1], color=color, alpha=0.9, capsize=3)
        ax.bar(x[index] + width / 2, other[index][0], width, yerr=other[index][1], color=color, alpha=0.45,
               hatch='//', edgecolor=color, capsize=3)
    if random_reference is not None:
        ax.axhline(random_reference['mean_reward'], color='gray', linestyle='--', linewidth=1, label='random (5x5)')
    ax.bar(0, 0, color='gray', alpha=0.9, label='5x5, stochastic (main study)')
    ax.bar(0, 0, color='gray', alpha=0.45, hatch='//', edgecolor='gray', label=setting_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Evaluation team reward per episode')
    ax.set_title(f'Policies trained on 5x5: main evaluation vs {setting_label}')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def comparison_table(groups, filename, setting_label):
    """Markdown table with mean Â± std across training seeds for both settings."""
    lines = [
        f'| Variant | 5x5, stochastic | {setting_label} | Ratio |',
        '|---|---|---|---|',
    ]
    for group in groups:
        base_mean, base_std = rp._mean_std(rp.evaluation_rewards_by_training_seed(group))
        other_values = _seed_means(group, filename)
        if not other_values:
            lines.append(f"| {group['label']} | {base_mean:.1f} Â± {base_std:.1f} | â€“ | â€“ |")
            continue
        other_mean, other_std = rp._mean_std(other_values)
        ratio = other_mean / base_mean if base_mean else float('nan')
        lines.append(
            f"| {group['label']} | {base_mean:.1f} Â± {base_std:.1f} | "
            f"{other_mean:.1f} Â± {other_std:.1f} | {ratio:.2f} |"
        )
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Grid-size and greedy evaluations of the final policies.')
    parser.add_argument('environment', choices=('hunt', 'harvest', 'escalation'))
    parser.add_argument('--grid-size', type=int, default=TRAINING_GRID,
                        help=f'Evaluate on an NxN grid (default: {TRAINING_GRID}, the training grid).')
    parser.add_argument('--deterministic', action='store_true', help='Greedy (argmax) actions instead of sampling.')
    parser.add_argument('--output-root', default='final_results')
    parser.add_argument('--eval-seeds', default=DEFAULT_EVAL_SEEDS)
    parser.add_argument('--episodes-per-seed', type=int, default=10)
    parser.add_argument('--skip-evaluation', action='store_true', help='Only redraw the plot/table.')
    args = parser.parse_args(argv)
    if args.grid_size == TRAINING_GRID and not args.deterministic:
        parser.error('Nothing to compare: pass --grid-size N (N != 5) and/or --deterministic.')

    output_root = PROJECT_ROOT / args.output_root
    if not args.skip_evaluation:
        _evaluate_missing(args.environment, output_root, args.grid_size, args.deterministic, args)

    suffix, setting_label = _setting(args.grid_size, args.deterministic)
    filename = f'evaluation_results_{suffix}.json'
    groups = _load_groups_with_paths(output_root, args.environment)
    colors = rp.color_map(groups)
    figures = output_root / 'figures' / args.environment
    fig = plot_comparison(groups, colors, filename, setting_label,
                          rp.load_random_reference(output_root, args.environment))
    rp.save_fig(fig, figures, f'14_evaluation_{suffix}')
    table = comparison_table(groups, filename, setting_label)
    (figures / f'evaluation_{suffix}.md').write_text(table + '\n', encoding='utf-8')
    print(table)
    print(f'[done] figure and table written to {figures}')


if __name__ == '__main__':
    main()
