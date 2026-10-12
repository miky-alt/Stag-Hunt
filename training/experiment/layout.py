"""Experiment identity and on-disk layout of one training run."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from training.config import TrainingConfig


def build_experiment_id(config: TrainingConfig, requested_name=None):
    """Readable prefix plus a hash of all non-seed settings, so equal configs share a suffix."""
    config_payload = json.dumps(
        {key: value for key, value in vars(config).items() if key != 'seed'},
        sort_keys=True,
        separators=(',', ':'),
    )
    config_hash = hashlib.sha1(config_payload.encode('utf-8')).hexdigest()[:8]
    if requested_name:
        return f"{requested_name}_{config_hash}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"experiment_{timestamp}_{config_hash}"


def load_config_file(path):
    """Reads a JSON file of TrainingConfig overrides, rejecting unknown fields."""
    overrides = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(overrides, dict):
        raise ValueError(f'Config file {path} must contain a JSON object.')
    unknown = sorted(set(overrides) - set(TrainingConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f'Unknown TrainingConfig fields in {path}: {unknown}')
    return overrides


def _layout(experiment_dir, experiment_id):
    return {
        'experiment_id': experiment_id,
        'root': experiment_dir,
        'checkpoints': experiment_dir / 'checkpoints',
        'training_logs': experiment_dir / 'training_logs',
    }


def create_experiment_paths(
    config: TrainingConfig,
    experiment_name=None,
    root_dir=None,
    resume=False,
    experiments_root='./experiments',
):
    """Creates a fresh run directory (with metadata) or resolves an existing one for resume."""
    experiments_root = Path(experiments_root)
    experiment_id = build_experiment_id(config, experiment_name)
    experiment_dir = Path(root_dir) if root_dir is not None else experiments_root / experiment_id

    if resume:
        # Prefer an exact directory, otherwise accept one unambiguous name match.
        exact_dir = experiments_root / experiment_name if root_dir is None else Path(root_dir)
        if exact_dir.exists():
            experiment_dir = exact_dir
        else:
            matching_dirs = list(experiments_root.glob(f'{experiment_name}_*'))
            if len(matching_dirs) == 1:
                experiment_dir = matching_dirs[0]
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        return _layout(experiment_dir, experiment_dir.name)

    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment directory already exists: {experiment_dir}. Choose a different --experiment-name."
        )
    paths = _layout(experiment_dir, experiment_id)
    for key, path in paths.items():
        if key != 'experiment_id':
            path.mkdir(parents=True, exist_ok=True)
    write_run_metadata(experiment_dir, experiment_id, config)
    return paths


def write_run_metadata(experiment_dir, experiment_id, config):
    """Writes metadata.json and a README describing the run layout."""
    metadata = {
        'experiment_id': experiment_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'hyperparameters': vars(config),
    }
    (experiment_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (experiment_dir / 'README.md').write_text(
        f"# Experiment `{experiment_id}`\n\n"
        "This directory contains the complete outputs for one isolated training run.\n\n"
        "- Hyperparameters: `metadata.json`\n"
        "- Checkpoints: `checkpoints/`\n"
        "- TensorBoard logs: `training_logs/`\n"
        "- Evaluation results: `evaluation_results.json`\n",
        encoding='utf-8',
    )


def write_group_metadata(group_dir, group_id, config, seeds):
    """Writes the metadata of a multi-seed training group."""
    (group_dir / 'metadata.json').write_text(json.dumps({
        'experiment_id': group_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'type': 'multi_seed_training_group',
        'seeds': seeds,
        'hyperparameters': vars(config),
    }, indent=2), encoding='utf-8')
    (group_dir / 'README.md').write_text(
        f'# Training group `{group_id}`\n\n'
        f'This experiment aggregates independent training runs for seeds: {seeds}.\n\n'
        '- Per-seed artifacts: `runs/seed_<seed>/`\n'
        '- Aggregate results: `training_results.json`\n'
        '- Group metadata: `metadata.json`\n',
        encoding='utf-8',
    )
