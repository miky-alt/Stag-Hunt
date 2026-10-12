"""Command-line entry point: `python -m training.train --experiment-name NAME [--config FILE] ...`."""
import os
# Force compatibility with Keras 2 to avoid internal TensorFlow errors.
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
from dataclasses import replace
from pathlib import Path

from agents.networks import VALID_NETWORK_INITS
from agents.ppo_custom import MultiAgentPPO
from envs.registry import ENVIRONMENT_IDS
from training.config import TrainingConfig
from training.experiment import load_config_file
from training.trainer import train_agent, train_seed_group

# Legacy metadata fields that no longer exist on TrainingConfig.
_OBSOLETE_HYPERPARAMETERS = (
    'convergence_reward_rate', 'convergence_max_rate_std',
    'early_stopping_reward', 'early_stopping_window', 'early_stopping_min_stags',
)


def build_parser():
    parser = argparse.ArgumentParser(description='Train a PPO agent in an isolated experiment directory.')
    parser.add_argument('--experiment-name', required=True,
                        help='Readable experiment prefix. Hyperparameter hash and timestamp are added automatically.')
    parser.add_argument('--resume-experiment', action='store_true',
                        help='Resume the existing experiment directory instead of creating a new one.')
    parser.add_argument('--config', help='JSON file with TrainingConfig overrides. Explicit CLI flags take precedence.')
    parser.add_argument('--output-root', default='./experiments',
                        help='Directory that will contain the experiment folder (default: ./experiments).')
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, help='Seed for one training run.')
    seed_group.add_argument('--seeds', help='Comma-separated seeds for independent sequential training runs.')

    parser.add_argument('--num-iterations', type=int)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--num-parallel-envs', type=int)
    parser.add_argument('--environment', choices=sorted(ENVIRONMENT_IDS))
    parser.add_argument('--handcrafted-features', action='store_true', default=None)
    parser.add_argument('--stag-follow-curriculum', dest='stag_follow_curriculum_enabled',
                        action='store_true', default=None)
    parser.add_argument('--stag-follow-curriculum-steps', type=int)
    parser.add_argument('--stag-follow-curriculum-mode', choices=('linear', 'step'))
    parser.add_argument('--disable-stag-following', action='store_true', default=None)
    parser.add_argument('--run-away-after-maul', dest='run_away_after_maul', action='store_true', default=None,
                        help='Hunt: respawn the stag after a mauling (default: it stays on its cell).')
    parser.add_argument('--agent-architecture', choices=sorted(MultiAgentPPO.VALID_ARCHITECTURES))
    parser.add_argument('--network-init', choices=VALID_NETWORK_INITS)
    parser.add_argument('--critic-observation', choices=('local', 'joint'))
    parser.add_argument('--reward-sharing-coef', type=float)
    parser.add_argument('--minibatch-size', type=int)
    parser.add_argument('--ppo-epochs', type=int)
    parser.add_argument('--curriculum-threshold', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--map-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--learning-rate-end', type=float)
    parser.add_argument('--clip-epsilon', type=float)
    parser.add_argument('--value-loss-coef', type=float)
    parser.add_argument('--value-normalization', action='store_true', default=None)
    parser.add_argument('--value-clipping', action='store_true', default=None)
    parser.add_argument('--value-clip-epsilon', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--gae-lambda', type=float)
    parser.add_argument('--entropy-start', type=float)
    parser.add_argument('--entropy-end', type=float)
    parser.add_argument('--entropy-decay-steps', type=int)
    parser.add_argument('--disable-curriculum', action='store_true', default=None)
    parser.add_argument('--convergence-custom-return-threshold', type=float)
    parser.add_argument('--convergence-window', type=int)
    parser.add_argument('--convergence-max-custom-return-std', type=float)
    return parser


def _resumed_config(parser, experiments_root, experiment_name):
    """Rebuilds the TrainingConfig saved in the metadata of an existing experiment."""
    candidates = list(experiments_root.glob(f'{experiment_name}_*/metadata.json'))
    exact = experiments_root / experiment_name / 'metadata.json'
    if exact.exists():
        candidates = [exact]
    if len(candidates) != 1:
        parser.error(f'Expected exactly one existing experiment matching {experiment_name}_*, found {len(candidates)}.')
    saved = json.loads(candidates[0].read_text(encoding='utf-8'))['hyperparameters'].copy()
    for field in _OBSOLETE_HYPERPARAMETERS:
        saved.pop(field, None)
    return replace(TrainingConfig(), **saved)


def resolve_config(parser, args):
    """Precedence: CLI flags > --config file > resumed metadata > TrainingConfig defaults."""
    experiments_root = Path(args.output_root)
    config = _resumed_config(parser, experiments_root, args.experiment_name) if args.resume_experiment else TrainingConfig()
    if args.config:
        config = replace(config, **load_config_file(args.config))
    overrides = {
        field: getattr(args, field, None)
        for field in TrainingConfig.__dataclass_fields__
        if getattr(args, field, None) is not None
    }
    if args.disable_curriculum:
        overrides['curriculum_enabled'] = False
    if args.disable_stag_following:
        overrides['stag_follows'] = False
    return replace(config, **overrides)


def parse_seeds(parser, args, config):
    if args.seeds is None:
        return [config.seed]
    try:
        seeds = [int(seed.strip()) for seed in args.seeds.split(',') if seed.strip()]
    except ValueError as error:
        parser.error(f'--seeds must be a comma-separated list of integers: {error}')
    if not seeds:
        parser.error('--seeds must contain at least one integer.')
    if len(seeds) != len(set(seeds)):
        parser.error('--seeds must not contain duplicate values.')
    return seeds


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve_config(parser, args)
    seeds = parse_seeds(parser, args, config)
    if len(seeds) == 1:
        train_agent(
            experiment_name=args.experiment_name,
            config=replace(config, seed=seeds[0]),
            resume=args.resume_experiment,
            experiments_root=Path(args.output_root),
        )
    else:
        train_seed_group(args.experiment_name, config, seeds, Path(args.output_root))


if __name__ == "__main__":
    main()
