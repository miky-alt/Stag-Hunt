import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf

from evaluate.evaluate import _sample_uniform_actions


def test_uniform_random_actions_are_valid_discrete_actions():
    time_step = type('TimeStep', (), {
        'observation': tf.zeros((2, 31), dtype=tf.float32),
    })()
    action_spec = type('ActionSpec', (), {'minimum': 0, 'maximum': 4})()

    actions = _sample_uniform_actions(time_step, action_spec)

    assert actions.shape == (2,)
    assert actions.dtype == tf.int32
    values = actions.numpy()
    assert np.all(values >= 0)
    assert np.all(values <= 4)