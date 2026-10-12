import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_agents.specs import tensor_spec

from training.config import TrainingConfig
from training.agent.factory import build_agent


def _build(architecture):
    """Builds a test agent using the requested sharing architecture."""
    # Define common per-agent specs so only the sharing topology varies.
    observation_spec = tensor_spec.TensorSpec((31,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 4)
    return build_agent(
        TrainingConfig(agent_architecture=architecture),
        observation_spec,
        action_spec,
    )


def test_shared_shared_uses_one_actor_and_one_critic():
    """Checks that shared_shared creates one actor and one critic."""
    # Build the shared topology and verify both networks have a single instance.
    agent = _build('shared_shared')

    assert len(agent.trainers) == 1
    assert agent.actors[0] is agent.actor
    assert agent.critics[0] is agent.critic


def test_separate_actors_shared_critic_shares_only_the_critic():
    """Checks that agents get distinct actors and one shared critic."""
    # Build the mixed topology and compare actor and critic object identities.
    agent = _build('separate_actors_shared_critic')

    assert agent.actors[0] is not agent.actors[1]
    assert agent.critics[0] is agent.critics[1]


def test_shared_actor_separate_critics_shares_only_the_actor():
    """Checks that one actor is shared while critics remain distinct."""
    agent = _build('shared_actor_separate_critics')

    assert agent.actors[0] is agent.actors[1]
    assert agent.critics[0] is not agent.critics[1]


def test_separate_actors_separate_critics_shares_nothing():
    """Checks that this architecture creates distinct actors and critics."""
    # Build the fully separate topology and verify neither network is shared.
    agent = _build('separate_actors_separate_critics')

    assert agent.actors[0] is not agent.actors[1]
    assert agent.critics[0] is not agent.critics[1]


def _layer_init_name(layer):
    """Returns the initializer class name for a dense layer's kernel."""
    return type(layer.kernel_initializer).__name__


def test_default_network_init_is_orthogonal_tanh():
    """Checks that default hidden and output layers use orthogonal initialization."""
    # Inspect hidden activation and kernel initializers on the default network.
    agent = _build('shared_shared')

    hidden = agent.actor._hidden_layers[0]
    assert hidden.activation is tf.keras.activations.tanh
    assert _layer_init_name(hidden) == 'Orthogonal'
    assert _layer_init_name(agent.actor._action_logits) == 'Orthogonal'


def test_legacy_network_init_restores_relu_he_normal_glorot():
    """Checks that legacy mode restores its previous activations and initializers."""
    # Build the legacy model, then inspect each hidden and output layer initializer.
    observation_spec = tensor_spec.TensorSpec((31,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 4)
    agent = build_agent(
        TrainingConfig(network_init='legacy'),
        observation_spec,
        action_spec,
    )

    for net, output in ((agent.actor, agent.actor._action_logits), (agent.critic, agent.critic._value_output)):
        for hidden in net._hidden_layers:
            assert hidden.activation is tf.keras.activations.relu
            assert _layer_init_name(hidden) == 'HeNormal'
        assert _layer_init_name(output) == 'GlorotUniform'


def test_unknown_network_init_is_rejected():
    """Checks that unsupported network initialization names raise an error."""
    # Supply valid specs and confirm the unsupported initializer fails at build time.
    observation_spec = tensor_spec.TensorSpec((31,), tf.float32)
    action_spec = tensor_spec.BoundedTensorSpec((), tf.int32, 0, 4)
    try:
        build_agent(TrainingConfig(network_init='bogus'), observation_spec, action_spec)
    except ValueError as error:
        assert 'network_init' in str(error)
    else:
        raise AssertionError('expected ValueError for unknown network_init')