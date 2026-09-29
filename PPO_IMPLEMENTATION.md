# PPO Implementation Status

This document records the implementation choices and the changes made to the
custom PPO trainer. It is intended as an experiment and methodology log, not
as a code reference.

## Environment and training structure

- The environment is a two-agent Stag Hunt with a shared categorical policy.
- Agent and critic networks are separate.
- Training uses reproducible seeds and supports independent multi-seed runs.
- Each training iteration collects a fixed rollout before updating the policy.
- The default rollout contains 260 steps and is retained in full.
- The rollout is divided into shuffled minibatches of 65 steps.
- Four PPO epochs are performed over the minibatches.

## PPO targets and updates

- Generalized Advantage Estimation is used with fixed discount and smoothing
  parameters.
- Advantages and return targets are computed once per rollout.
- The same targets and old action log-probabilities are reused throughout the
  PPO epochs for that rollout.
- Advantages are normalized before optimization.
- The policy uses the clipped surrogate objective.
- The value function is trained with a mean-squared error objective.
- Entropy regularization encourages exploration and is decayed during training.
- Global gradient norm clipping is enabled.

## Network and optimizer choices

- Hidden layers use `tanh` activations, matching the standard PPO reference
  configuration used by the implementation-details report.
- Hidden layers use orthogonal weight initialization with gain `sqrt(2)` and
  zero biases.
- The actor output uses orthogonal initialization with gain `0.01`.
- The critic output uses orthogonal initialization with gain `1.0`.
- The learning rate is linearly annealed from its configured initial value to
  its configured final value over the training iterations.

## Diagnostics and evaluation

TensorBoard records the main PPO diagnostics: policy loss, value loss, entropy,
approximate KL divergence, clipping fraction, gradient norm, learning rate,
returns, and behavior statistics. Training results also preserve the final
per-iteration diagnostic values and the per-agent convergence information.

Convergence speed is measured independently for each agent using reward per
step. An agent is considered stable after the configured reward threshold is
met for a consecutive window while the variation remains below the configured
limit. Overall convergence is reported only after both agents have converged.

## Implementation history

- Added seed handling and isolated experiment artifacts.
- Added multi-seed training and aggregate mean/std reporting.
- Added training and evaluation heatmap persistence and analysis.
- Added reward-based convergence analysis.
- Corrected rollout retention so the complete rollout is available to PPO.
- Added shuffled minibatch updates and fixed rollout-level GAE targets.
- Added PPO-style initialization, activation, learning-rate annealing, and
  diagnostics.