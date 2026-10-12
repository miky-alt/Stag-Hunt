"""Regenerate every figure and summary table of the study from final_results/.

    python scripts/make_report_figures.py            # all games
    python scripts/make_report_figures.py hunt       # one game

Outputs go to final_results/figures/<game>/ (PNG + summary_table.md) and
final_results/figures/00_training_overview.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import report_plots as rp  # noqa: E402

GAMES = ('hunt', 'harvest', 'escalation')
# Exploratory variants that are not part of the report.
EXCLUDED_VARIANTS = ('stationary',)


def make_game_figures(root, environment, window=10):
    spec = rp.ENVIRONMENT_SPECS[environment]
    figures = root / 'figures' / environment
    groups = [
        group for group in rp.load_groups(root, environment)
        if not any(token in group['label'] for token in EXCLUDED_VARIANTS)
    ]
    if not groups:
        print(f'[{environment}] no completed runs under {root}')
        return
    colors = rp.color_map(groups)
    random_reference = rp.load_random_reference(root, environment)

    outputs = [
        ('01_training_team_return', rp.plot_team_return(groups, colors, window=window)),
        ('02_training_agent_returns', rp.plot_agent_returns(groups, window=window)),
        ('03_training_ppo_diagnostics', rp.plot_ppo_diagnostics(groups, colors, window=window)),
        ('04_training_behavior', rp.plot_training_behavior(groups, colors, spec, window=window)),
        ('05_training_final_return', rp.plot_final_training_return(groups, colors)),
        ('07_evaluation_team_reward', rp.plot_evaluation_rewards(groups, colors, random_reference)),
        ('08_evaluation_agent_reward', rp.plot_evaluation_agent_rewards(groups)),
        ('09_train_vs_eval', rp.plot_train_vs_eval(groups, colors)),
        ('10_evaluation_behavior_events', rp.plot_behavior_events(groups, spec, colors)),
        ('11_evaluation_team_events', rp.plot_team_events(groups, spec, colors)),
        ('12_evaluation_heatmaps', rp.plot_heatmaps(groups, source='events')),
        ('13_training_heatmaps', rp.plot_heatmaps(groups, source='training')),
    ]
    for name, fig in outputs:
        if fig is not None:
            rp.save_fig(fig, figures, name)
            plt.close(fig)
    table = rp.summary_table(groups, spec, random_reference)
    (figures / 'summary_table.md').write_text(table + '\n', encoding='utf-8')
    print(f'[{environment}] {len(groups)} variants -> {figures}')


def main(argv=None):
    games = tuple(argv) if argv else GAMES
    root = PROJECT_ROOT / 'final_results'
    for environment in games:
        make_game_figures(root, environment)
    groups = {env: rp.load_groups(root, env) for env in GAMES}
    if any(groups.values()):
        fig = rp.plot_training_overview(groups, window=10, exclude=EXCLUDED_VARIANTS)
        rp.save_fig(fig, root / 'figures', '00_training_overview')
        plt.close(fig)
        print(f"[overview] -> {root / 'figures' / '00_training_overview.png'}")


if __name__ == '__main__':
    main(sys.argv[1:])
