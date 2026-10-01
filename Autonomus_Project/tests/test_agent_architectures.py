import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_agents.specs import tensor_spec

from training.config import TrainingConfig
from training.train import _build_agent


def _build(architecture):
    observation_spec = tensor_spec.TensorSpec((31,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 4)
    return _build_agent(
        TrainingConfig(agent_architecture=architecture),
        observation_spec,
        action_spec,
    )


def test_shared_shared_uses_one_actor_and_one_critic():
    agent = _build('shared_shared')

    assert len(agent.trainers) == 1
    assert agent.actors[0] is agent.actor
    assert agent.critics[0] is agent.critic


def test_separate_actors_shared_critic_shares_only_the_critic():
    agent = _build('separate_actors_shared_critic')

    assert agent.actors[0] is not agent.actors[1]
    assert agent.critics[0] is agent.critics[1]


def test_separate_actors_separate_critics_shares_nothing():
    agent = _build('separate_actors_separate_critics')

    assert agent.actors[0] is not agent.actors[1]
    assert agent.critics[0] is not agent.critics[1]