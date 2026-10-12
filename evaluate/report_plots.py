"""Shared loading and plotting helpers for the per-environment report notebooks.

All three notebooks (hunt, harvest, escalation) call the same functions so that
figure layout, colors, ordering and statistics are identical across games.
Only ``ENVIRONMENT_SPECS`` carries environment-specific labels and event names.
"""
import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Environment-specific vocabulary; everything else in this module is generic.
ENVIRONMENT_SPECS = {
    'hunt': {
        'title': 'Stag Hunt – Hunt',
        # (metric prefix in behavior_event_evaluation_results, display label)
        'events': [
            ('stags_caught', 'Stag captures'),
            ('plants_eaten', 'Plants eaten'),
            ('maulings_sustained', 'Maulings sustained'),
        ],
        'team_events': [],
        # Training counters are reward-threshold based and only valid for Hunt.
        'training_behavior_metrics': [
            ('stags_caught', 'Captures per iteration'),
            ('plants_eaten', 'Plants per iteration'),
            ('maulings_sustained', 'Maulings per iteration'),
        ],
        'curriculum_metric': 'stag_follow_probability',
    },
    'harvest': {
        'title': 'Stag Hunt – Harvest',
        'events': [
            ('plants_harvested_young', 'Young plants harvested'),
            ('plants_harvested_mature', 'Mature plants harvested'),
        ],
        'team_events': [
            ('plants_harvested_young', 'Unique young plants (team)'),
            ('plants_harvested_mature', 'Unique mature plants (team)'),
        ],
        'training_behavior_metrics': [],
        'curriculum_metric': None,
    },
    'escalation': {
        'title': 'Stag Hunt – Escalation',
        'events': [
            ('reciprocal_chase_plus_ones', '+1 reciprocal mark events'),
            ('streak_break_punishments', 'Streak-break punishments (count)'),
            ('streak_break_punishment_amount', 'Streak-break punishments (magnitude)'),
        ],
        'team_events': [],
        'training_behavior_metrics': [],
        'curriculum_metric': None,
    },
}

PPO_DIAGNOSTICS = [
    ('loss', 'Total loss'),
    ('policy_loss', 'Policy loss'),
    ('value_loss', 'Value loss'),
    ('policy_entropy', 'Policy entropy'),
    ('entropy', 'Entropy coefficient'),
    ('approx_kl', 'Approximate KL'),
    ('clip_fraction', 'Clip fraction'),
    ('gradient_norm', 'Gradient norm'),
    ('learning_rate', 'Learning rate'),
]

AGENT_COLORS = {'a': '#1f77b4', 'b': '#ff7f0e'}
FIGSIZE_WIDE = (12, 5)
FIGSIZE_GRID = (14, 8)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def variant_label(experiment_id, environment):
    """Turns 'final_hunt_02_reward_sharing_1.0_1bb4c40e' into '02 reward_sharing_1.0'."""
    name = re.sub(rf'^final_{environment}_', '', experiment_id)
    name = re.sub(r'_[0-9a-f]{8}$', '', name)
    return name.replace('_', ' ', 1)


def load_groups(root, environment):
    """Loads every completed ablation group of one environment, sorted by variant."""
    root = Path(root)
    groups = []
    for group_dir in sorted(root.glob(f'final_{environment}_*')):
        if not (group_dir / 'metadata.json').exists():
            continue
        metadata = _read_json(group_dir / 'metadata.json')
        training_path = group_dir / 'training_results.json'
        runs = []
        for run_dir in sorted((group_dir / 'runs').glob('seed_*')):
            run = {'seed': int(run_dir.name.split('_')[-1])}
            for key, filename in (
                ('training', 'training_results.json'),
                ('evaluation', 'evaluation_results.json'),
                ('events', 'behavior_event_evaluation_results.json'),
            ):
                path = run_dir / filename
                run[key] = _read_json(path) if path.exists() else None
            if run['events'] is None and run['evaluation'] is not None:
                run['events'] = run['evaluation']
            runs.append(run)
        groups.append({
            'experiment_id': metadata['experiment_id'],
            'label': variant_label(metadata['experiment_id'], environment),
            'hyperparameters': metadata['hyperparameters'],
            'training': _read_json(training_path) if training_path.exists() else None,
            'runs': runs,
        })
    return groups


def load_random_reference(root, environment):
    path = Path(root) / f'random_reference_{environment}.json'
    return _read_json(path) if path.exists() else None


def color_map(groups):
    """Stable color per variant: same index -> same color in every figure."""
    cmap = plt.get_cmap('tab10' if len(groups) <= 10 else 'tab20')
    return {group['label']: cmap(index % cmap.N) for index, group in enumerate(groups)}


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _mean_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def smooth(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode='valid')


def evaluation_rewards_by_training_seed(group, key='mean_reward'):
    """One value per training seed: the evaluation mean over 3 eval seeds x 10 episodes."""
    return [run['evaluation'][key] for run in group['runs'] if run['evaluation']]


def pooled_episode_rewards(group, key='reward'):
    return [
        episode[key]
        for run in group['runs'] if run['evaluation']
        for episode in run['evaluation']['episodes']
    ]


def event_values_by_training_seed(group, event_name):
    """Per training seed: mean over evaluation seeds of the per-episode event mean."""
    values = []
    for run in group['runs']:
        if not run['events']:
            continue
        per_seed = [
            seed['behavior_events'][event_name]['mean_per_episode']
            for seed in run['events']['by_seed']
            if event_name in seed['behavior_events']
        ]
        if per_seed:
            values.append(float(np.mean(per_seed)))
    return values


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, figures_dir, name):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f'{name}.png', dpi=150, bbox_inches='tight')


def plot_training_curve(groups, metric, ax, colors, ylabel=None, window=10, show_std=True):
    """Mean (± std across training seeds) of one metric by iteration, all variants overlaid."""
    for group in groups:
        training = group['training']
        if not training or metric not in training['metrics']:
            continue
        series = training['metrics'][metric]
        mean = smooth(series['mean_by_iteration'], window)
        iterations = np.arange(1, len(mean) + 1)
        ax.plot(iterations, mean, label=group['label'], color=colors[group['label']], linewidth=1.5)
        if show_std:
            std = smooth(series['std_by_iteration'], window)
            ax.fill_between(iterations, mean - std, mean + std, color=colors[group['label']], alpha=0.12)
    ax.set_xlabel('PPO iteration')
    ax.set_ylabel(ylabel or metric)
    ax.grid(alpha=0.3)


def plot_team_return(groups, colors, window=10, title='Team return during training'):
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plot_training_curve(groups, 'return', ax, colors, ylabel='Mean return per episode', window=window)
    ax.set_title(f'{title} (mean ± std across 3 seeds, smoothing {window})')
    ax.legend(fontsize=8, ncol=2)
    return fig


def variant_name(label):
    """'02 reward_sharing_1.0' -> 'reward_sharing_1.0' (the numeric prefix differs per game)."""
    return label.split(' ', 1)[1] if ' ' in label else label


def plot_training_overview(groups_by_environment, window=10, exclude=()):
    """One panel per game with the team return of every variant; colors are shared by variant name."""
    names = []
    for groups in groups_by_environment.values():
        for group in groups:
            name = variant_name(group['label'])
            if name not in names and not any(token in name for token in exclude):
                names.append(name)
    cmap = plt.get_cmap('tab10' if len(names) <= 10 else 'tab20')
    name_colors = {name: cmap(index % cmap.N) for index, name in enumerate(names)}

    fig, axes = plt.subplots(1, len(groups_by_environment), figsize=(3.6 * len(groups_by_environment), 2.3))
    for ax, (environment, groups) in zip(np.atleast_1d(axes), groups_by_environment.items()):
        kept = [group for group in groups if variant_name(group['label']) in name_colors]
        colors = {group['label']: name_colors[variant_name(group['label'])] for group in kept}
        plot_training_curve(kept, 'return', ax, colors, ylabel='Team return per episode', window=window)
        ax.set_title(environment.capitalize(), fontsize=10)
        if ax is not np.atleast_1d(axes)[0]:
            ax.set_ylabel('')
    handles = [plt.Line2D([], [], color=color, linewidth=2) for color in name_colors.values()]
    fig.legend(handles, [name.replace('_', ' ') for name in name_colors], loc='lower center',
               ncol=min(6, len(names)), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig


def plot_agent_returns(groups, window=10):
    """One panel per variant: agent A vs agent B training return."""
    cols = 2 if len(groups) <= 6 else 3
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, group in zip(axes, groups):
        training = group['training']
        if not training:
            continue
        for agent in ('a', 'b'):
            series = training['metrics'][f'agent_{agent}_return']
            mean = smooth(series['mean_by_iteration'], window)
            std = smooth(series['std_by_iteration'], window)
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, color=AGENT_COLORS[agent], label=f'Agent {agent.upper()}')
            ax.fill_between(x, mean - std, mean + std, color=AGENT_COLORS[agent], alpha=0.15)
        ax.set_title(group['label'], fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[len(groups):]:
        ax.axis('off')
    axes[0].legend(fontsize=8)
    fig.suptitle('Agent return during training (mean ± std across 3 seeds)')
    fig.tight_layout()
    return fig


def plot_ppo_diagnostics(groups, colors, window=10, metrics=PPO_DIAGNOSTICS):
    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_GRID)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        plot_training_curve(groups, metric, ax, colors, ylabel=label, window=window, show_std=False)
        ax.set_title(label, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), fontsize=8)
    fig.suptitle('PPO diagnostics during training (mean across 3 seeds)')
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig


def plot_training_behavior(groups, colors, spec, window=10):
    """Hunt only: per-iteration stag/plant/mauling counters and the curriculum schedule."""
    metrics = list(spec['training_behavior_metrics'])
    if spec['curriculum_metric']:
        metrics.append((spec['curriculum_metric'], 'Stag-follow probability'))
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4))
    for ax, (metric, label) in zip(np.atleast_1d(axes), metrics):
        plot_training_curve(groups, metric, ax, colors, ylabel=label, window=window, show_std=False)
        ax.set_title(label, fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle('Behavior observed in training rollouts (mean across 3 seeds)')
    fig.tight_layout()
    return fig


def plot_final_training_return(groups, colors):
    """Final training return per variant: bar = mean, dots = individual training seeds."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for index, group in enumerate(groups):
        training = group['training']
        if not training:
            continue
        by_seed = list(training['metrics']['return']['final_by_seed'].values())
        mean, std = _mean_std(by_seed)
        ax.bar(index, mean, yerr=std, color=colors[group['label']], alpha=0.8, capsize=4)
        ax.scatter([index] * len(by_seed), by_seed, color='black', s=14, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([group['label'] for group in groups], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Final training return')
    ax.set_title('Final training return (mean ± std across 3 seeds; dots = seeds)')
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_evaluation_rewards(groups, colors, random_reference=None):
    """Team reward: bars = mean ± std across training seeds, box = pooled episodes."""
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(15, 5))
    labels = [group['label'] for group in groups]
    for index, group in enumerate(groups):
        values = evaluation_rewards_by_training_seed(group)
        mean, std = _mean_std(values)
        ax_bar.bar(index, mean, yerr=std, color=colors[group['label']], alpha=0.8, capsize=4)
        ax_bar.scatter([index] * len(values), values, color='black', s=14, zorder=3)
    pooled = [pooled_episode_rewards(group) for group in groups]
    box = ax_box.boxplot(pooled, patch_artist=True, showfliers=True)
    for patch, group in zip(box['boxes'], groups):
        patch.set_facecolor(colors[group['label']])
        patch.set_alpha(0.7)
    for ax in (ax_bar, ax_box):
        if random_reference:
            ax.axhline(random_reference['mean_reward'], color='red', linestyle='--',
                       linewidth=1, label='Random policy')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel('Team reward per episode')
    ax_bar.set_xticks(range(len(groups)))
    ax_bar.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax_box.set_xticks(range(1, len(groups) + 1))
    ax_box.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax_bar.set_title('Mean ± std across training seeds (dots = seeds)')
    ax_box.set_title('Episode distribution (3 training seeds × 3 eval seeds × 10 episodes)')
    if random_reference:
        ax_bar.legend(fontsize=8)
    fig.suptitle('Team reward during evaluation')
    fig.tight_layout()
    return fig


def plot_evaluation_agent_rewards(groups):
    """Grouped bars: agent A vs agent B evaluation reward, mean ± std across training seeds."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    width = 0.38
    x = np.arange(len(groups))
    for offset, agent in ((-width / 2, 'a'), (width / 2, 'b')):
        means, stds = [], []
        for group in groups:
            mean, std = _mean_std(evaluation_rewards_by_training_seed(group, f'mean_agent_{agent}_reward'))
            means.append(mean)
            stds.append(std)
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=AGENT_COLORS[agent],
               alpha=0.85, label=f'Agent {agent.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels([group['label'] for group in groups], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Agent reward per episode')
    ax.set_title('Agent reward during evaluation (mean ± std across training seeds)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig


def plot_train_vs_eval(groups, colors):
    """Does the final training return predict the evaluation reward? One point per training seed."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for group in groups:
        training = group['training']
        if not training:
            continue
        final_by_seed = training['metrics']['return']['final_by_seed']
        for run in group['runs']:
            if not run['evaluation']:
                continue
            ax.scatter(final_by_seed[str(run['seed'])], run['evaluation']['mean_reward'],
                       color=colors[group['label']], label=group['label'], s=40)
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(limits, limits, color='grey', linestyle=':', linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7)
    ax.set_xlabel('Final training return (per seed)')
    ax.set_ylabel('Mean evaluation reward (per seed)')
    ax.set_title('Training vs evaluation (each point = one training seed)')
    ax.grid(alpha=0.3)
    return fig


def plot_behavior_events(groups, spec, colors):
    """One panel per event: agent A vs agent B mean per episode, ± std across training seeds."""
    events = spec['events']
    fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 4.5))
    width = 0.38
    x = np.arange(len(groups))
    for ax, (prefix, label) in zip(np.atleast_1d(axes), events):
        for offset, agent in ((-width / 2, 'a'), (width / 2, 'b')):
            means, stds = [], []
            for group in groups:
                mean, std = _mean_std(event_values_by_training_seed(group, f'{prefix}_agent_{agent}'))
                means.append(mean)
                stds.append(std)
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=AGENT_COLORS[agent],
                   alpha=0.85, label=f'Agent {agent.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels([group['label'] for group in groups], rotation=45, ha='right', fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Events per episode')
        ax.grid(axis='y', alpha=0.3)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=8, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle('Behavioral events during evaluation (mean ± std across training seeds)')
    fig.tight_layout()
    return fig


def plot_team_events(groups, spec, colors):
    """Team-level unique events (used by Harvest to avoid double counting mature plants)."""
    events = spec['team_events']
    if not events:
        return None
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    width = 0.8 / len(events)
    x = np.arange(len(groups))
    for offset_index, (name, label) in enumerate(events):
        offset = (offset_index - (len(events) - 1) / 2) * width
        means, stds = [], []
        for group in groups:
            mean, std = _mean_std(event_values_by_training_seed(group, name))
            means.append(mean)
            stds.append(std)
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([group['label'] for group in groups], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Events per episode')
    ax.set_title('Team events during evaluation (mean ± std across training seeds)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig


def plot_heatmaps(groups, source='events'):
    """Occupancy heatmaps (agent A top row, agent B bottom row), normalized per variant.

    source='events' uses the evaluation heatmaps saved in
    behavior_event_evaluation_results.json; source='training' uses training rollouts.
    """
    fig, axes = plt.subplots(2, len(groups), figsize=(2.2 * len(groups), 5.0))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(wspace=0.15, hspace=0.05, right=0.9)
    for column, group in enumerate(groups):
        maps = {'a': None, 'b': None}
        for run in group['runs']:
            payload = run[source]
            if not payload:
                continue
            container = payload['behavior'] if source == 'events' else payload
            for agent in maps:
                grid = np.asarray(container[f'heatmap_agent_{agent}'], dtype=float)
                maps[agent] = grid if maps[agent] is None else maps[agent] + grid
        for row, agent in enumerate(('a', 'b')):
            ax = axes[row, column]
            if maps[agent] is None:
                ax.axis('off')
                continue
            grid = maps[agent] / maps[agent].sum()
            image = ax.imshow(grid.T, cmap='viridis', origin='upper', vmin=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title('\n'.join(textwrap.wrap(group['label'].replace('_', ' '), 16)), fontsize=7)
            if column == 0:
                ax.set_ylabel(f'Agent {agent.upper()}')
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, label='Fraction of steps in cell')
    label = 'evaluation' if source == 'events' else 'training'
    fig.suptitle(f'Grid occupancy during {label} (sum of 3 seeds, normalized)', y=1.02)
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def summary_table(groups, spec, random_reference=None):
    """Markdown table with training, evaluation and per-agent event statistics."""
    event_columns = [
        (f'{prefix}_agent_{agent}', f'{label} {agent.upper()}')
        for prefix, label in spec['events'] for agent in ('a', 'b')
    ] + [(name, label) for name, label in spec['team_events']]
    header = ['Variant', 'Final train return', 'Eval reward (seed)', 'Eval A', 'Eval B']
    header += [label for _, label in event_columns]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]

    def fmt(values):
        mean, std = _mean_std(values)
        return 'n/a' if np.isnan(mean) else f'{mean:.1f} ± {std:.1f}'

    if random_reference:
        row = ['policy random', '–', f"{random_reference['mean_reward']:.1f} ± {random_reference['std_reward']:.1f}"]
        row += ['–'] * (len(header) - 3)
        lines.append('| ' + ' | '.join(row) + ' |')
    for group in groups:
        training = group['training']
        row = [
            group['label'],
            fmt(list(training['metrics']['return']['final_by_seed'].values())) if training else 'n/a',
            fmt(evaluation_rewards_by_training_seed(group)),
            fmt(evaluation_rewards_by_training_seed(group, 'mean_agent_a_reward')),
            fmt(evaluation_rewards_by_training_seed(group, 'mean_agent_b_reward')),
        ]
        row += [fmt(event_values_by_training_seed(group, name)) for name, _ in event_columns]
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)
