"""Shared driver for the final ablation study of one environment.

For every config in configs/final/<environment>: train the configured seeds,
evaluate each trained seed on fixed evaluation seeds, then refresh the summary
tables. Completed stages are skipped, so a run can be interrupted and resumed.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / 'configs' / 'final'
DEFAULT_TRAINING_SEEDS = '42,43,44'
DEFAULT_EVAL_SEEDS = '101,202,303'


def run_logged(command, log_path):
    """Run a command, streaming its output to the console and a log file."""
    # Prepare a consistent child-process environment and log destination.
    env = {**os.environ, 'TF_USE_LEGACY_KERAS': '1', 'TF_ENABLE_ONEDNN_OPTS': '0'}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log_file:
        # Stream merged output once to both the terminal and persistent log.
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        process.wait()
    return process.returncode


def _find_group_dir(output_root, experiment_name):
    """Finds the output directory matching an experiment name prefix."""
    matches = sorted(output_root.glob(f'{experiment_name}_*'))
    return matches[0] if matches else None


def _run_dirs(group_dir):
    """Per-seed run folders; a single-seed training writes directly into the group folder."""
    return sorted((group_dir / 'runs').glob('seed_*')) or [group_dir]


def _evaluate_random_reference(environment, output_root, log_dir, args):
    """Evaluate independent uniform random actions once per environment (reward floor)."""
    # Reuse completed baselines; otherwise find metadata from a trained run.
    target = output_root / f'random_reference_{environment}.json'
    if target.exists():
        print(f'[skip] random reference already complete: {target.name}')
        return
    # Any trained run of this environment provides the metadata needed to build the env.
    run_dirs = [
        run_dir
        for group_dir in sorted(output_root.glob(f'final_{environment}_*'))
        for run_dir in _run_dirs(group_dir)
        if (run_dir / 'checkpoints').exists()
    ]
    if not run_dirs:
        return
    run_dir = run_dirs[0]
    run_name = run_dir.relative_to(output_root).as_posix()
    print(f'[eval] random uniform reference for {environment} (via {run_name})')
    # Evaluate the random policy with the same seeds and episode count as the study.
    return_code = run_logged(
        [
            sys.executable, '-m', 'evaluate.evaluate',
            '--experiments-root', str(output_root),
            '--experiment', run_name,
            '--policy', 'random_uniform',
            '--seeds', args.eval_seeds,
            '--episodes-per-seed', str(args.episodes_per_seed),
            '--output-file', target.name,
        ],
        log_dir / f'eval_random_reference_{environment}.log',
    )
    if return_code != 0:
        raise RuntimeError(f'Random reference evaluation failed: {environment}')
    shutil.move(run_dir / target.name, target)


def run_environment(environment, argv=None):
    """Runs training, evaluation, and summary generation for one environment."""
    # Parse study options and resolve the config/output directories for this game.
    parser = argparse.ArgumentParser(
        description=f'Run the final {environment} ablations (train + evaluate + summary).'
    )
    parser.add_argument('--output-root', default='final_results')
    parser.add_argument('--seeds', default=DEFAULT_TRAINING_SEEDS, help='Training seeds.')
    parser.add_argument('--eval-seeds', default=DEFAULT_EVAL_SEEDS, help='Evaluation seeds.')
    parser.add_argument('--episodes-per-seed', type=int, default=10)
    parser.add_argument(
        '--only',
        nargs='*',
        help='Config file stems to run (default: every config of the environment).',
    )
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument(
        '--train-arg',
        action='append',
        default=[],
        help='Extra argument forwarded verbatim to training.train (repeatable).',
    )
    args = parser.parse_args(argv)

    output_root = PROJECT_ROOT / args.output_root
    log_dir = output_root / 'logs'
    config_paths = sorted((CONFIG_ROOT / environment).glob('*.json'))
    if args.only:
        config_paths = [path for path in config_paths if path.stem in set(args.only)]
    if not config_paths:
        parser.error(f'No config files found for {environment}.')

    # Process each ablation, skipping completed stages and refusing partial groups.
    for config_path in config_paths:
        experiment_name = f'final_{environment}_{config_path.stem}'
        group_dir = _find_group_dir(output_root, experiment_name)

        if not args.skip_training:
            if group_dir is not None and (group_dir / 'training_results.json').exists():
                print(f'[skip] training already complete: {group_dir.name}')
            else:
                if group_dir is not None:
                    raise RuntimeError(
                        f'Incomplete training folder found: {group_dir}. Remove it before re-running.'
                    )
                print(f'[train] {experiment_name}')
                return_code = run_logged(
                    [
                        sys.executable, '-m', 'training.train',
                        '--experiment-name', experiment_name,
                        '--config', str(config_path),
                        '--seeds', args.seeds,
                        '--output-root', str(output_root),
                        *args.train_arg,
                    ],
                    log_dir / f'train_{experiment_name}.log',
                )
                if return_code != 0:
                    raise RuntimeError(f'Training failed: {experiment_name}')
                group_dir = _find_group_dir(output_root, experiment_name)

        # Evaluate every completed seed run unless evaluation was explicitly skipped.
        if args.skip_evaluation or group_dir is None:
            continue
        for run_dir in _run_dirs(group_dir):
            run_name = run_dir.relative_to(output_root).as_posix()
            if (run_dir / 'evaluation_results.json').exists():
                print(f'[skip] evaluation already complete: {run_name}')
                continue
            print(f'[eval] {run_name}')
            return_code = run_logged(
                [
                    sys.executable, '-m', 'evaluate.evaluate',
                    '--experiments-root', str(output_root),
                    '--experiment', run_name,
                    '--seeds', args.eval_seeds,
                    '--episodes-per-seed', str(args.episodes_per_seed),
                ],
                log_dir / f'eval_{run_name.replace("/", "_")}.log',
            )
            if return_code != 0:
                raise RuntimeError(f'Evaluation failed: {run_name}')

    # Refresh the random baseline and aggregate summary after all selected configs.
    if not args.skip_evaluation:
        _evaluate_random_reference(environment, output_root, log_dir, args)

    print('[summary] writing summary.json and summary.md')
    subprocess.run(
        [sys.executable, '-m', 'evaluate.summarize_final_results', '--root', str(output_root)],
        cwd=PROJECT_ROOT,
        check=True,
    )
