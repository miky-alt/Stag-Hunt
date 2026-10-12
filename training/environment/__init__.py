"""Training environment construction and Hunt curricula."""
from training.environment.curriculum import (
    activate_curriculum,
    set_stag_follow_probability,
    stag_follow_probability,
)
from training.environment.factory import create_training_environment, environment_kwargs_for

__all__ = [
    'activate_curriculum',
    'create_training_environment',
    'environment_kwargs_for',
    'set_stag_follow_probability',
    'stag_follow_probability',
]
