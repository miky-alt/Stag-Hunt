import numpy as np
import tensorflow as tf

from training.ppo import mix_agent_rewards


def test_reward_sharing_zero_preserves_individual_rewards():
    """Checks that zero sharing leaves each agent's reward unchanged."""
    # Mix one known reward pair with sharing disabled and compare the result.
    rewards = tf.constant([[[2.0, 6.0]]])

    mixed = mix_agent_rewards(rewards, 0.0)

    np.testing.assert_allclose(mixed.numpy(), rewards.numpy())


def test_reward_sharing_half_assigns_the_mean_to_both_agents():
    """Checks that half sharing gives both agents the mean reward."""
    # Use an asymmetric pair so the equalized team reward is unambiguous.
    rewards = tf.constant([[[2.0, 6.0]]])

    mixed = mix_agent_rewards(rewards, 0.5)

    np.testing.assert_allclose(mixed.numpy(), [[[4.0, 4.0]]])


def test_reward_sharing_rejects_invalid_coefficients():
    """Checks that reward-sharing coefficients outside zero to one are rejected."""
    # Exercise values on both sides of the allowed coefficient interval.
    rewards = tf.constant([[[2.0, 6.0]]])

    for coefficient in (-0.1, 1.1):
        try:
            mix_agent_rewards(rewards, coefficient)
        except ValueError:
            continue
        raise AssertionError('Invalid reward-sharing coefficient was accepted.')