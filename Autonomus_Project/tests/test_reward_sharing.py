import numpy as np
import tensorflow as tf

from training.rollouts import _mix_agent_rewards


def test_reward_sharing_zero_preserves_individual_rewards():
    rewards = tf.constant([[[2.0, 6.0]]])

    mixed = _mix_agent_rewards(rewards, 0.0)

    np.testing.assert_allclose(mixed.numpy(), rewards.numpy())


def test_reward_sharing_half_assigns_the_mean_to_both_agents():
    rewards = tf.constant([[[2.0, 6.0]]])

    mixed = _mix_agent_rewards(rewards, 0.5)

    np.testing.assert_allclose(mixed.numpy(), [[[4.0, 4.0]]])


def test_reward_sharing_rejects_invalid_coefficients():
    rewards = tf.constant([[[2.0, 6.0]]])

    for coefficient in (-0.1, 1.1):
        try:
            _mix_agent_rewards(rewards, coefficient)
        except ValueError:
            continue
        raise AssertionError('Invalid reward-sharing coefficient was accepted.')