"""Builds the actor/critic trainers for each supported parameter-sharing architecture."""
import tensorflow as tf
from tf_agents.specs import tensor_spec

from agents.networks import ActorNetwork, CriticNetwork
from agents.ppo_custom import CustomPPO, MultiAgentPPO
from training.config import TrainingConfig

HIDDEN_SIZES = (128, 128)


def network_specs(tf_env, is_parallel):
    """Per-agent observation/action specs the networks are built against."""
    if not is_parallel:
        return tf_env.observation_spec(), tf_env.action_spec()
    # Parallel workers batch both agents together; networks see one agent at a time.
    observation_spec = tensor_spec.TensorSpec(
        shape=(tf_env.observation_spec().shape[-1],), dtype=tf.float32, name='observation'
    )
    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(), dtype=tf.int32, minimum=0, maximum=4, name='action'
    )
    return observation_spec, action_spec


def build_agent(config: TrainingConfig, observation_spec, action_spec):
    """Builds the MultiAgentPPO for the configured architecture and critic input."""
    def build_actor():
        return ActorNetwork(
            observation_spec, action_spec, hidden_sizes=HIDDEN_SIZES, init=config.network_init
        )

    if config.critic_observation not in {'local', 'joint'}:
        raise ValueError("critic_observation must be 'local' or 'joint'.")
    critic_spec = observation_spec
    critic_output_size = 1
    if config.critic_observation == 'joint':
        critic_spec = tensor_spec.TensorSpec(
            shape=(observation_spec.shape[-1] * 2,),
            dtype=observation_spec.dtype,
            name='joint_observation',
        )
        critic_output_size = 2

    def build_critic():
        return CriticNetwork(
            critic_spec, hidden_sizes=HIDDEN_SIZES, output_size=critic_output_size, init=config.network_init
        )

    architecture = config.agent_architecture
    if architecture == 'shared_shared':
        actors = [build_actor()]
        critics = [build_critic()]
    elif architecture == 'shared_actor_separate_critics':
        shared_actor = build_actor()
        actors = [shared_actor, shared_actor]
        critics = [build_critic(), build_critic()]
    elif architecture == 'separate_actors_shared_critic':
        actors = [build_actor(), build_actor()]
        shared_critic = build_critic()
        critics = [shared_critic, shared_critic]
    elif architecture == 'separate_actors_separate_critics':
        actors = [build_actor(), build_actor()]
        critics = [build_critic(), build_critic()]
    else:
        raise ValueError(
            f'Unknown agent architecture: {architecture}. '
            f'Use one of {sorted(MultiAgentPPO.VALID_ARCHITECTURES)}.'
        )

    # Build each unique network once, including networks shared by two trainers.
    for network in {id(net): net for net in actors + critics}.values():
        network.create_variables()

    trainers = [
        CustomPPO(
            actor_net=actor,
            critic_net=critic,
            lr=config.learning_rate,
            clip_epsilon=config.clip_epsilon,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_start,
            value_normalization=config.value_normalization,
            value_clipping=config.value_clipping,
            value_clip_epsilon=config.value_clip_epsilon,
        )
        for actor, critic in zip(actors, critics)
    ]
    # A shared critic must also share its running return statistics.
    if config.value_normalization and architecture in {'shared_shared', 'separate_actors_shared_critic'}:
        shared_normalizer = trainers[0].value_normalizer
        for trainer in trainers[1:]:
            trainer.value_normalizer = shared_normalizer
    return MultiAgentPPO(trainers, architecture)
