"""Behavioural event counters exposed by each game's `info` dictionary."""
import numpy as np

# info key -> flat label; per-agent integer counts.
EVENT_COUNT_FIELDS = {
    'hunt': {
        'stags_caught_by_agent': 'stags_caught',
        'plants_eaten_by_agent': 'plants_eaten',
        'maulings_sustained_by_agent': 'maulings_sustained',
    },
    'harvest': {
        'plants_harvested_young_by_agent': 'plants_harvested_young',
        'plants_harvested_mature_by_agent': 'plants_harvested_mature',
    },
    'escalation': {
        'reciprocal_chase_plus_ones_by_agent': 'reciprocal_chase_plus_ones',
        'streak_break_punishment_events_by_agent': 'streak_break_punishments',
    },
}
# info key -> flat label; per-agent float amounts.
EVENT_AMOUNT_FIELDS = {
    'escalation': {
        'streak_break_punishment_amount_by_agent': 'streak_break_punishment_amount',
    },
}
# Team-level scalar counters copied verbatim.
EVENT_SCALAR_FIELDS = {
    'harvest': ('plants_harvested_young', 'plants_harvested_mature'),
}


def behavior_event_names(environment):
    """Lists the flat event fields recorded for one game."""
    names = []
    for label in EVENT_COUNT_FIELDS.get(environment, {}).values():
        names.extend((f'{label}_agent_a', f'{label}_agent_b'))
    for label in EVENT_AMOUNT_FIELDS.get(environment, {}).values():
        names.extend((f'{label}_agent_a', f'{label}_agent_b'))
    names.extend(EVENT_SCALAR_FIELDS.get(environment, ()))
    return names


def extract_behavior_event_counts(environment, info):
    """Converts a game's info payload of one step into flat, JSON-friendly event counts."""
    events = {}
    for info_key, label in EVENT_COUNT_FIELDS.get(environment, {}).items():
        values = info.get(info_key, (0, 0))
        events[f'{label}_agent_a'] = int(values[0])
        events[f'{label}_agent_b'] = int(values[1])
    for info_key, label in EVENT_AMOUNT_FIELDS.get(environment, {}).items():
        values = info.get(info_key, (0.0, 0.0))
        events[f'{label}_agent_a'] = float(values[0])
        events[f'{label}_agent_b'] = float(values[1])
    for field in EVENT_SCALAR_FIELDS.get(environment, ()):
        events[field] = int(info.get(field, 0))
    return events


def summarize_behavior_events(episode_values):
    """Mean/std/total per event over the episodes of one evaluation seed."""
    return {
        name: {
            'mean_per_episode': float(np.mean(values)),
            'std_across_episodes': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            'total': float(np.sum(values)),
        }
        for name, values in episode_values.items()
    }
