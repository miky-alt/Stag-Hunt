import numpy as np

from evaluate.events import (
    extract_behavior_event_counts,
    summarize_behavior_events,
)
from evaluate.statistics import summarize_episode_rewards
from evaluate.summarize_final_results import _summarize_behavior_evaluation

def test_evaluation_statistics_are_computed_per_complete_episode_and_agent():
    """Checks that evaluation rewards are summarized per episode and agent."""
    # Provide complete paired episodes, then verify team and individual summaries.
    statistics = summarize_episode_rewards([
        [10.0, 2.0],
        [0.0, 4.0],
        [6.0, 8.0],
    ])

    assert np.isclose(statistics['mean_agent_a_reward'], 16.0 / 3.0)
    assert np.isclose(statistics['mean_agent_b_reward'], 14.0 / 3.0)
    assert np.isclose(statistics['mean_reward'], 10.0)
    assert statistics['std_agent_a_reward'] > 0.0
    assert statistics['std_agent_b_reward'] > 0.0


def test_behavior_event_counts_are_extracted_per_agent():
    """Preserves individual agent attribution and team-unique Harvest counts."""
    events = extract_behavior_event_counts('harvest', {
        'plants_harvested_young': 1,
        'plants_harvested_mature': 1,
        'plants_harvested_young_by_agent': [0, 1],
        'plants_harvested_mature_by_agent': [1, 0],
    })

    assert events == {
        'plants_harvested_young_agent_a': 0,
        'plants_harvested_young_agent_b': 1,
        'plants_harvested_mature_agent_a': 1,
        'plants_harvested_mature_agent_b': 0,
        'plants_harvested_young': 1,
        'plants_harvested_mature': 1,
    }


def test_hunt_and_escalation_events_keep_agent_attribution():
    """Maps game event payloads to explicit per-agent evaluation fields."""
    hunt_events = extract_behavior_event_counts('hunt', {
        'stags_caught_by_agent': [1, 1],
        'plants_eaten_by_agent': [2, 0],
        'maulings_sustained_by_agent': [0, 1],
    })
    escalation_events = extract_behavior_event_counts('escalation', {
        'reciprocal_chase_plus_ones_by_agent': [1, 1],
        'streak_break_punishment_events_by_agent': [1, 0],
        'streak_break_punishment_amount_by_agent': [2.5, 0.0],
    })

    assert hunt_events['stags_caught_agent_a'] == hunt_events['stags_caught_agent_b'] == 1
    assert hunt_events['plants_eaten_agent_a'] == 2
    assert hunt_events['maulings_sustained_agent_b'] == 1
    assert escalation_events['reciprocal_chase_plus_ones_agent_a'] == 1
    assert escalation_events['streak_break_punishments_agent_a'] == 1
    assert escalation_events['streak_break_punishment_amount_agent_a'] == 2.5


def test_behavior_events_summarize_evaluation_seed_variability():
    """Separates per-episode spread from variability between evaluation seeds."""
    per_seed = summarize_behavior_events({'event_agent_a': [1, 3, 5]})
    combined = _summarize_behavior_evaluation({
        'by_seed': [
            {'behavior_events': {'event_agent_a': per_seed['event_agent_a']}},
            {'behavior_events': {'event_agent_a': {
                'mean_per_episode': 5.0,
                'std_across_episodes': 0.0,
                'total': 15.0,
            }}},
        ]
    })

    assert per_seed['event_agent_a']['mean_per_episode'] == 3.0
    assert np.isclose(per_seed['event_agent_a']['std_across_episodes'], 2.0)
    assert combined['event_agent_a']['mean_per_episode_across_evaluation_seeds'] == 4.0
    assert np.isclose(combined['event_agent_a']['std_across_evaluation_seeds'], np.sqrt(2.0))