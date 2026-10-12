"""Command-line entry point: `python -m evaluate.evaluate --experiment DIR [--experiments-root ROOT] ...`."""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse

from evaluate.runner import DEFAULT_EVAL_SEEDS, POLICIES, evaluate_agent


def build_parser():
    parser = argparse.ArgumentParser(description='Evaluate one isolated training experiment.')
    parser.add_argument('--experiment', required=True, help='Experiment directory name under the experiments root.')
    parser.add_argument('--experiments-root', default='./experiments',
                        help='Directory containing the experiment folder (default: ./experiments).')
    parser.add_argument('--seeds', default=','.join(str(seed) for seed in DEFAULT_EVAL_SEEDS),
                        help='Comma-separated evaluation seeds.')
    parser.add_argument('--episodes-per-seed', '--episodes', dest='episodes_per_seed', type=int, default=10,
                        help='Number of evaluation episodes for each seed.')
    parser.add_argument('--render', action='store_true', help='Render the evaluation episodes.')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use argmax actions instead of sampling from the checkpoint policy.')
    parser.add_argument('--render-delay', type=float, default=0.3,
                        help='Delay in seconds between rendered steps (default: 0.3).')
    parser.add_argument('--checkpoint-step', type=int, help='Load this exact checkpoint step instead of the latest.')
    parser.add_argument('--policy', choices=POLICIES, default='checkpoint',
                        help='Evaluation policy: trained checkpoint or independent uniform random actions.')
    parser.add_argument('--output-file', default=None,
                        help='Result filename inside the experiment directory '
                             '(default: evaluation_results.json, or evaluation_results_grid<N>.json with --grid-size).')
    parser.add_argument('--grid-size', type=int,
                        help='Evaluate on an NxN grid instead of the training grid (generalization test).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    seeds = [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]
    output_file = args.output_file or (
        f'evaluation_results_grid{args.grid_size}.json' if args.grid_size is not None else 'evaluation_results.json'
    )
    evaluate_agent(
        args.experiment,
        episodes_per_seed=args.episodes_per_seed,
        seeds=seeds,
        render=args.render,
        policy=args.policy,
        deterministic=args.deterministic,
        render_delay=args.render_delay,
        output_filename=output_file,
        checkpoint_step=args.checkpoint_step,
        experiments_root=args.experiments_root,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
