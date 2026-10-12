"""Reward statistics over complete evaluation episodes."""
import numpy as np


def summarize_episode_rewards(agent_rewards):
    """Team and per-agent mean/std from an [episode, agent] reward matrix."""
    rewards = np.asarray(agent_rewards, dtype=np.float64)
    if rewards.ndim != 2 or rewards.shape[1] != 2 or rewards.shape[0] == 0:
        raise ValueError('Expected a non-empty [episode, agent] reward array.')
    team_rewards = np.sum(rewards, axis=1)

    def _std(values):
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        'mean_reward': float(np.mean(team_rewards)),
        'std_reward': _std(team_rewards),
        'mean_agent_a_reward': float(np.mean(rewards[:, 0])),
        'std_agent_a_reward': _std(rewards[:, 0]),
        'mean_agent_b_reward': float(np.mean(rewards[:, 1])),
        'std_agent_b_reward': _std(rewards[:, 1]),
    }
