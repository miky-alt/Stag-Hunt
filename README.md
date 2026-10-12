# Learning to Cooperate in Stag Hunt Gridworlds

An ablation study of PPO design choices for two-agent coordination in the three
[Gymnasium-Stag-Hunt](https://github.com/giorgiofranceschelli/Gymnasium-Stag-Hunt)
gridworlds (**Hunt**, **Harvest**, **Escalation**). Full write-up: [`report.pdf`](report.pdf).

## Overview

Stag Hunt games capture a core difficulty of multi-agent RL: the socially optimal
outcome requires every agent to commit to an action that pays off only if the partner
commits as well, so independent learners tend to settle on the safe but inefficient
equilibrium. Starting from a parameter-sharing independent-PPO baseline (one actor, one
local critic for both agents), the study ablates:

- **Prosocial reward mixing** `r_A' = (1-c) r_A + c r_B` with `c ∈ {0.5, 1}`;
- **Parameter sharing**: shared actor / separate critics, separate actors / shared critic,
  fully independent actors and critics;
- **Critic input**: local observation vs. joint observation (incl. MAPPO = separate
  actors + joint critic);
- **Handcrafted geometric features** (Manhattan distances and displacements);
- **A curriculum on stag mobility** (Hunt only).

29 variants (11 Hunt, 9 Harvest, 9 Escalation) are each trained with 3 seeds and
evaluated with a fixed seed-controlled protocol (3 evaluation seeds × 10 episodes,
100 steps), comparing team reward and exact per-agent behavioural event counters
(stag captures, plants eaten, maulings, young/mature harvests, reciprocal mark steps,
streak breaks) against a uniform-random floor. Two robustness checks reuse the same
protocol: greedy (argmax) execution and evaluation on an unseen 7×7 grid.

Main findings: in Hunt the baseline never captures the stag, while full reward
swapping (`c = 1`) yields ≈45 captures per episode and separate actors recover more
than half of that gain without touching the reward; in Escalation reward mixing,
independent actors and MAPPO all reach the same plateau of ≈95 reciprocal steps,
whereas every shared-actor variant falls short and collapses under greedy execution;
in Harvest, where no dilemma exists, geometric features dominate.

### Code layout

```text
agents/          actor/critic networks, PPO trainers (CustomPPO, MultiAgentPPO)
envs/            TF-Agents wrappers around the games, registry of game IDs
training/        train.py (CLI) -> trainer.py (orchestration)
  agent/           network construction, checkpoint layout, LR/entropy schedules
  environment/     single/parallel environment factory, Hunt curricula
  rollout/         experience collection and on-policy buffer
  ppo/             reward mixing, GAE, minibatch PPO update
  experiment/      experiment ids, directory layout, JSON configs
  monitoring/      TensorBoard logging, convergence criterion, result files
evaluate/        evaluate.py (CLI) -> runner.py; events, policies, statistics,
                 summarize_final_results.py, report_plots.py, analyze_heatmap.py
configs/final/   one JSON per variant and game (the 29 configurations of the report)
scripts/         study drivers: run_final_<game>.py, grid_generalization.py,
                 make_report_figures.py
gymnasium_stag_hunt/  vendored fork of the environments, extended with per-agent
                 event counters in `info` and a stag-mobility probability
tests/           pytest suite
```

## Installation

Python 3.11 is required (TensorFlow 2.21 / tf-agents 0.18).

```bash
git clone <this repository> stag-hunt-ppo
cd stag-hunt-ppo
python -m venv venv
# Windows:  venv\Scripts\activate      Linux/macOS:  source venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the vendored gymnasium_stag_hunt environments
python -m pytest tests -q   # sanity check (≈30 s)
```

All commands below are run from the repository root with the virtual environment active.
Training runs on CPU; one variant (3 seeds × 1000 iterations) takes roughly 20–40 minutes
on a laptop.

## Reproducing the results of the report

### 1. Train and evaluate all variants

```bash
python scripts/run_final_hunt.py
python scripts/run_final_harvest.py
python scripts/run_final_escalation.py
```

For each JSON in `configs/final/<game>/` this trains seeds 42, 43, 44, evaluates every
seed on evaluation seeds 101, 202, 303 (10 episodes each), computes the random-policy
floor, and writes `final_results/summary.md` / `summary.json`. Completed stages are
skipped, so an interrupted run can simply be relaunched. To run a subset:

```bash
python scripts/run_final_hunt.py --only 01_baseline 02_reward_sharing_1.0 10_mappo
```

Per-run artefacts: `final_results/final_<game>_<variant>_<hash>/runs/seed_<seed>/`
(`metadata.json`, `checkpoints/`, `training_logs/` for TensorBoard,
`training_results.json`, `evaluation_results.json`).

### 2. Robustness checks (Table 1, last two columns; Section 5.4)

```bash
python scripts/grid_generalization.py <game> --deterministic   # greedy actions, 5x5
python scripts/grid_generalization.py <game> --grid-size 7     # sampling, unseen 7x7 grid
```

Results are written next to the main evaluation as `evaluation_results_greedy.json` /
`evaluation_results_grid7.json`, plus a comparison table and bar plot in
`final_results/figures/<game>/`.

### 3. Figures and tables

```bash
python scripts/make_report_figures.py
```

Produces, for every game, training curves, PPO diagnostics, evaluation reward
distributions, behavioural event bars, heatmaps and `summary_table.md` under
`final_results/figures/<game>/`, plus the cross-game training overview
(`final_results/figures/00_training_overview.png`, Figure 1 of the report).

### Single runs

```bash
# one seed, settings from a JSON config (any TrainingConfig field can be overridden by a flag)
python -m training.train --experiment-name my_run --config configs/final/hunt/02_reward_sharing_1.0.json
# three seeds under one group directory
python -m training.train --experiment-name my_run --config configs/final/hunt/01_baseline.json --seeds 42,43,44
# resume an interrupted run
python -m training.train --experiment-name my_run --resume-experiment

# evaluate a run (sampling), greedy, or the uniform-random floor
python -m evaluate.evaluate --experiment <experiment_dir> --seeds 101,202,303 --episodes-per-seed 10
python -m evaluate.evaluate --experiment <experiment_dir> --deterministic --output-file evaluation_results_greedy.json
python -m evaluate.evaluate --experiment <experiment_dir> --policy random_uniform --output-file random.json
# watch the agents play in the PyGame grid window (as in the upstream Gym-Stag-Hunt demos)
python -m evaluate.evaluate --experiment <experiment_dir> --seeds 101 --episodes-per-seed 3 --render --render-delay 0.2
# inspect behaviour totals and position heatmaps
python -m evaluate.analyze_heatmap --experiment <experiment_dir> --source evaluation
```

`<experiment_dir>` is the directory name printed at start-up (e.g.
`my_run_1bb4c40e` or `my_run_1bb4c40e/runs/seed_42`), relative to
`--experiments-root` (default `./experiments`; the study scripts use `final_results`).
Experiment directories embed a hash of the non-seed configuration and are never reused.

## License

MIT (see `LICENSE`). The environments are a fork of Gymnasium-Stag-Hunt by
Giorgio Franceschelli, itself derived from Gym-Stag-Hunt by David Nesterov-Rappoport.
