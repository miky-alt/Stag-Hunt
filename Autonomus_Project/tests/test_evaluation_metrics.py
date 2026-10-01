import numpy as np

from evaluate.evaluate import _summarize_episode_rewards

def test_evaluation_statistics_are_computed_per_complete_episode_and_agent():
    statistics = _summarize_episode_rewards([
        [10.0, 2.0],
        [0.0, 4.0],
        [6.0, 8.0],
    ])

    assert np.isclose(statistics['mean_agent_a_reward'], 16.0 / 3.0)
    assert np.isclose(statistics['mean_agent_b_reward'], 14.0 / 3.0)
    assert np.isclose(statistics['mean_reward'], 10.0)
    assert statistics['std_agent_a_reward'] > 0.0
    assert statistics['std_agent_b_reward'] > 0.0