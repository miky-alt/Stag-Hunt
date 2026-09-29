import argparse
import json
from pathlib import Path

import numpy as np


class HeatmapAnalyzer:
    """Print behavior totals and position heatmaps from an evaluation or training result file."""

    def __init__(self, results: dict, unit_count: int, unit_label: str):
        self.unit_count = unit_count
        self.unit_label = unit_label
        self.total_stags = results['total_stags_caught']
        self.total_plants = results['total_plants_eaten']
        self.total_maulings = results['total_maulings_sustained']
        self.heatmap_a = np.array(results['heatmap_agent_a'], dtype=int)
        self.heatmap_b = np.array(results['heatmap_agent_b'], dtype=int)

    def print_report(self):
        average_stags = self.total_stags / self.unit_count if self.unit_count else 0
        average_plants = self.total_plants / self.unit_count if self.unit_count else 0
        average_maulings = self.total_maulings / self.unit_count if self.unit_count else 0

        print("\n==================================================")
        print("          GLOBAL BEHAVIOR SUMMARY       ")
        print("==================================================")
        print(f"Total stags caught: {self.total_stags} (Average/{self.unit_label}: {average_stags:.1f})")
        print(f"Total plants eaten: {self.total_plants} (Average/{self.unit_label}: {average_plants:.1f})")
        print(f"Total maulings sustained: {self.total_maulings} (Average/{self.unit_label}: {average_maulings:.1f})")

        self._print_heatmap_section("AGENT A", self.heatmap_a)
        self._print_heatmap_section("AGENT B", self.heatmap_b)
        print("==================================================\n")

    def _print_heatmap_section(self, label, heatmap):
        print("\n==================================================")
        print(f"    GLOBAL HEATMAP {label} (Percentages)        ")
        print("==================================================")
        total = np.sum(heatmap)
        if total > 0:
            self._print_grid((heatmap / total) * 100)
        else:
            print(f"[!] No movement data recorded for {label.lower()}.")

    def _print_grid(self, percentage_grid):
        for row in percentage_grid:
            formatted_row = [f"{value:5.1f}%" for value in row]
            print(" | ".join(formatted_row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the behavior recorded during evaluation or training.')
    parser.add_argument('--experiment', required=True, help='Experiment directory name under ./experiments.')
    parser.add_argument(
        '--source',
        choices=['evaluation', 'training'],
        default='evaluation',
        help='Read behavior/heatmap data from evaluation_results.json or training_results.json.'
    )
    args = parser.parse_args()

    experiment_dir = Path('./experiments') / args.experiment
    metadata_path = experiment_dir / 'metadata.json'
    results_filename = 'evaluation_results.json' if args.source == 'evaluation' else 'training_results.json'
    results_path = experiment_dir / results_filename

    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    if not results_path.exists():
        producer = 'evaluate.py' if args.source == 'evaluation' else 'training.train'
        raise FileNotFoundError(
            f"{results_filename} not found: {results_path}. Run {producer} for this experiment first."
        )

    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    results = json.loads(results_path.read_text(encoding='utf-8'))

    print(f"Experiment: {metadata['experiment_id']}")
    print(f"Hyperparameters: {json.dumps(metadata['hyperparameters'], sort_keys=True)}")

    if args.source == 'evaluation':
        behavior = results.get('behavior')
        if not behavior:
            raise ValueError(
                "No 'behavior' data found in evaluation_results.json. "
                "Re-run evaluation with the current evaluate.py to record it."
            )
        analyzer = HeatmapAnalyzer(behavior, unit_count=len(results.get('episodes', [])), unit_label='episode')
    else:
        analyzer = HeatmapAnalyzer(results, unit_count=results.get('iterations', 0), unit_label='iteration')

    analyzer.print_report()
