"""Aggregate final ablation results into summary.json and summary.md."""
import argparse
import json
from pathlib import Path

import numpy as np

ABLATION_FIELDS = (
    'agent_architecture',
    'critic_observation',
    'reward_sharing_coef',
    'handcrafted_features',
    'stag_follow_curriculum_enabled',
    'stag_follow_curriculum_steps',
    'stag_follow_curriculum_mode',
)


def _mean_std(values):
    """Returns the mean and sample standard deviation, or None for empty data."""
    # Normalize inputs and define empty/single-sample behavior explicitly.
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), std


def _summarize_behavior_evaluation(event_evaluation):
    """Summarizes event means across evaluation seeds for one training run."""
    by_seed = event_evaluation.get('by_seed', [])
    event_names = sorted({
        name
        for entry in by_seed
        for name in entry.get('behavior_events', {})
    })
    summary = {}
    for name in event_names:
        seed_means = [
            entry['behavior_events'][name]['mean_per_episode']
            for entry in by_seed
            if name in entry.get('behavior_events', {})
        ]
        seed_episode_stds = [
            entry['behavior_events'][name]['std_across_episodes']
            for entry in by_seed
            if name in entry.get('behavior_events', {})
        ]
        mean, std = _mean_std(seed_means)
        summary[name] = {
            'mean_per_episode_across_evaluation_seeds': mean,
            'std_across_evaluation_seeds': std,
            'mean_episode_std_across_evaluation_seeds': (
                float(np.mean(seed_episode_stds)) if seed_episode_stds else None
            ),
            'evaluation_seeds': len(seed_means),
        }
    return summary


def summarize_group(group_dir):
    """Combines one experiment group's training and evaluation results."""
    # Load group metadata and optional aggregate training output.
    metadata = json.loads((group_dir / 'metadata.json').read_text(encoding='utf-8'))
    hyperparameters = metadata['hyperparameters']
    training_results_path = group_dir / 'training_results.json'
    training_results = (
        json.loads(training_results_path.read_text(encoding='utf-8'))
        if training_results_path.exists() else None
    )

    # Collect completed per-seed evaluations while skipping unfinished runs.
    seed_evaluations = []
    run_dirs = sorted((group_dir / 'runs').glob('seed_*')) or [group_dir]
    for run_dir in run_dirs:
        evaluation_path = run_dir / 'evaluation_results.json'
        if not evaluation_path.exists():
            continue
        evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        event_path = run_dir / 'behavior_event_evaluation_results.json'
        event_evaluation = (
            json.loads(event_path.read_text(encoding='utf-8'))
            if event_path.exists() else None
        )
        seed_evaluations.append({
            'training_seed': (
                int(run_dir.name.split('_')[-1]) if run_dir != group_dir
                else hyperparameters.get('seed')
            ),
            'checkpoint_step': evaluation['checkpoint_step'],
            'mean_reward': evaluation['mean_reward'],
            'std_reward': evaluation['std_reward'],
            'mean_agent_a_reward': evaluation['mean_agent_a_reward'],
            'mean_agent_b_reward': evaluation['mean_agent_b_reward'],
            'episode_rewards': [episode['reward'] for episode in evaluation['episodes']],
            'behavior': {
                key: evaluation['behavior'][key]
                for key in (
                    'total_stags_caught', 'total_plants_eaten', 'total_maulings_sustained'
                ) if key in evaluation.get('behavior', {})
            },
            'behavior_events': (
                _summarize_behavior_evaluation(event_evaluation)
                if event_evaluation else {}
            ),
        })

    # Compute both seed-level and pooled-episode statistics for comparison.
    seed_means = [entry['mean_reward'] for entry in seed_evaluations]
    pooled_episodes = [reward for entry in seed_evaluations for reward in entry['episode_rewards']]
    mean_across_seeds, std_across_seeds = _mean_std(seed_means)
    pooled_mean, pooled_std = _mean_std(pooled_episodes)
    behavior_event_names = sorted({
        name
        for entry in seed_evaluations
        for name in entry['behavior_events']
    })
    behavior_events = {}
    for name in behavior_event_names:
        training_seed_means = [
            entry['behavior_events'][name]['mean_per_episode_across_evaluation_seeds']
            for entry in seed_evaluations
            if name in entry['behavior_events']
        ]
        training_seed_eval_stds = [
            entry['behavior_events'][name]['std_across_evaluation_seeds']
            for entry in seed_evaluations
            if name in entry['behavior_events']
        ]
        mean, std = _mean_std(training_seed_means)
        behavior_events[name] = {
            'mean_per_episode_across_training_seeds': mean,
            'std_across_training_seeds': std,
            'mean_std_across_evaluation_seeds': (
                float(np.mean(training_seed_eval_stds))
                if training_seed_eval_stds else None
            ),
            'training_seeds': len(training_seed_means),
        }

    # Read convergence/training metrics when this group has completed training.
    final_return = None
    if training_results and 'metrics' in training_results:
        final_return = training_results['metrics']['return']

    return {
        'experiment_id': metadata['experiment_id'],
        'environment': hyperparameters['environment'],
        'ablation': {field: hyperparameters.get(field) for field in ABLATION_FIELDS},
        'training_seeds': metadata.get('seeds', [hyperparameters.get('seed')]),
        'training': {
            'iterations': training_results['iterations'] if training_results else None,
            'final_return_mean': final_return['final_mean'] if final_return else None,
            'final_return_std': final_return['final_std'] if final_return else None,
            'first_stable_window_iteration_mean': (
                training_results.get('first_stable_window_iteration_mean')
                if training_results else None
            ),
            'stable_seeds': training_results.get('stable_seeds') if training_results else None,
        },
        'evaluation': {
            'evaluated_training_seeds': len(seed_evaluations),
            'episodes': len(pooled_episodes),
            'mean_reward_across_training_seeds': mean_across_seeds,
            'std_reward_across_training_seeds': std_across_seeds,
            'pooled_mean_reward': pooled_mean,
            'pooled_std_reward': pooled_std,
            'behavior_events': behavior_events,
            'by_training_seed': seed_evaluations,
        },
    }


def _fmt(mean, std):
    """Formats a mean and standard deviation for the Markdown report."""
    if mean is None:
        return 'n/a'
    return f'{mean:.1f} ± {std:.1f}'


def load_random_references(root):
    """Loads saved random-policy evaluation summaries by environment."""
    # Index baseline files by environment name for the final report.
    references = {}
    for path in sorted(root.glob('random_reference_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        references[path.stem.replace('random_reference_', '')] = {
            'mean_reward': data['mean_reward'],
            'std_reward': data['std_reward'],
            'episodes': len(data['episodes']),
        }
    return references


def write_markdown(summaries, random_references, path):
    """Writes experiment and random-baseline summaries as Markdown tables."""
    # Start with the protocol description, then render one section per game.
    lines = ['# Final ablation results', '']
    lines.append(
        'Evaluation: mean ± std of the per-training-seed mean team reward '
        '(3 training seeds × 3 evaluation seeds × 10 episodes). '
        'Pooled: statistics over all evaluation episodes. '
        'The random row is an independent uniform-random policy evaluated with the same protocol.'
    )
    for environment in ('hunt', 'harvest', 'escalation'):
        rows = [summary for summary in summaries if summary['environment'] == environment]
        if not rows and environment not in random_references:
            continue
        lines += ['', f'## {environment}', '']
        lines.append(
            '| Variant | Architecture | Critic obs | Reward sharing | Handcrafted | Stag curriculum | '
            'Train final return | Eval reward (seed mean ± std) | Eval reward (pooled) | Eval episodes |'
        )
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        if environment in random_references:
            reference = random_references[environment]
            lines.append(
                f"| random uniform policy | – | – | – | – | – | – | – | "
                f"{_fmt(reference['mean_reward'], reference['std_reward'])} | {reference['episodes']} |"
            )
        # Format each ablation row using its saved hyperparameters and metrics.
        for summary in rows:
            ablation = summary['ablation']
            curriculum = (
                f"{ablation['stag_follow_curriculum_steps']} it. ({ablation.get('stag_follow_curriculum_mode', 'linear')})"
                if ablation.get('stag_follow_curriculum_enabled') else 'off'
            )
            evaluation = summary['evaluation']
            training = summary['training']
            lines.append(
                f"| {summary['experiment_id']} | {ablation['agent_architecture']} | "
                f"{ablation.get('critic_observation', 'local')} | {ablation['reward_sharing_coef']} | "
                f"{'yes' if ablation.get('handcrafted_features') else 'no'} | {curriculum} | "
                f"{_fmt(training['final_return_mean'], training['final_return_std'])} | "
                f"{_fmt(evaluation['mean_reward_across_training_seeds'], evaluation['std_reward_across_training_seeds'])} | "
                f"{_fmt(evaluation['pooled_mean_reward'], evaluation['pooled_std_reward'])} | "
                f"{evaluation['episodes']} |"
            )
        event_rows = [
            (summary, name, stats)
            for summary in rows
            for name, stats in summary['evaluation']['behavior_events'].items()
        ]
        if event_rows:
            lines += ['', f'### Behavior events: {environment}', '']
            lines.append(
                '| Variant | Event | Mean per episode across training seeds | '
                'Mean std across evaluation seeds |'
            )
            lines.append('|---|---|---|---|')
            for summary, name, stats in event_rows:
                lines.append(
                    f"| {summary['experiment_id']} | {name} | "
                    f"{_fmt(stats['mean_per_episode_across_training_seeds'], stats['std_across_training_seeds'])} | "
                    f"{stats['mean_std_across_evaluation_seeds']:.2f} |"
                )
    # Persist a stable newline-terminated report for review and versioning.
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    """Reads final experiment outputs and writes JSON and Markdown summaries."""
    # Discover completed groups, load random baselines, and write both reports.
    parser = argparse.ArgumentParser(description='Summarize final ablation results.')
    parser.add_argument('--root', default='final_results', help='Directory containing the training groups.')
    args = parser.parse_args()
    root = Path(args.root)
    summaries = [
        summarize_group(group_dir)
        for group_dir in sorted(root.iterdir())
        if group_dir.is_dir() and group_dir.name.startswith('final_')
        and (group_dir / 'metadata.json').exists()
    ]
    random_references = load_random_references(root)
    (root / 'summary.json').write_text(
        json.dumps({'random_references': random_references, 'groups': summaries}, indent=2),
        encoding='utf-8',
    )
    write_markdown(summaries, random_references, root / 'summary.md')
    print(f'Wrote {root / "summary.json"} and {root / "summary.md"} ({len(summaries)} groups).')


if __name__ == '__main__':
    main()
