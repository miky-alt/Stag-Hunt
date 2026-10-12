"""Gymnasium IDs of the three Stag Hunt games, keyed by the short names used in configs."""

ENVIRONMENT_IDS = {
    'hunt': 'StagHunt-Hunt-v0',
    'harvest': 'StagHunt-Harvest-v0',
    'escalation': 'StagHunt-Escalation-v0',
}


def resolve_environment_id(environment):
    """Returns the Gymnasium ID for a short environment name, or raises a clear error."""
    try:
        return ENVIRONMENT_IDS[environment]
    except KeyError as error:
        raise ValueError(
            f"Unknown environment '{environment}'. Choose one of {sorted(ENVIRONMENT_IDS)}."
        ) from error
