"""Evaluation policies: restored actor networks (sampling or greedy) and the uniform-random floor."""
from pathlib import Path

import numpy as np
import tensorflow as tf
from tf_agents.utils import common

from agents.networks import ActorNetwork
from training.agent.checkpoint import actor_checkpoint_items

SHARED_ACTOR_ARCHITECTURES = {'shared_shared', 'shared_actor_separate_critics'}


def build_actor_networks(hyperparameters, observation_spec, action_spec):
    """One actor for shared-actor architectures, two otherwise; mirrors training construction."""
    architecture = hyperparameters.get('agent_architecture', 'shared_shared')
    actor_count = 1 if architecture in SHARED_ACTOR_ARCHITECTURES else 2
    return [
        ActorNetwork(
            input_tensor_spec=observation_spec,
            output_tensor_spec=action_spec,
            hidden_sizes=(128, 128),
            init=hyperparameters.get('network_init', 'orthogonal'),
        )
        for _ in range(actor_count)
    ]


def restore_actors(actor_nets, checkpoint_dir, train_step_counter, checkpoint_step=None):
    """Restores the latest checkpoint, or an exact step, into the actors only."""
    items = actor_checkpoint_items(actor_nets, train_step_counter)
    if checkpoint_step is None:
        status = common.Checkpointer(ckpt_dir=str(checkpoint_dir), **items).initialize_or_restore()
    else:
        checkpoint_path = Path(checkpoint_dir) / f'ckpt-{checkpoint_step}'
        if not Path(f'{checkpoint_path}.index').exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        status = tf.train.Checkpoint(**items).restore(str(checkpoint_path))
    status.expect_partial()


def _pick(action_distribution, deterministic):
    return tf.argmax(action_distribution.logits, axis=-1) if deterministic else action_distribution.sample()


def actor_actions(actor_nets, observation, deterministic):
    """Actions of both agents from one shared actor or one actor per agent."""
    if len(actor_nets) == 1:
        action_distribution, _ = actor_nets[0](observation)
        return tf.cast(_pick(action_distribution, deterministic), tf.int32)
    actions = []
    for agent_index, actor_net in enumerate(actor_nets):
        action_distribution, _ = actor_net(observation[agent_index:agent_index + 1])
        actions.append(tf.cast(_pick(action_distribution, deterministic)[0], tf.int32))
    return tf.stack(actions)


def sample_uniform_actions(time_step, action_spec):
    """One independent uniform discrete action per agent (reward floor)."""
    batch_size = int(np.asarray(time_step.observation.numpy()).shape[0])
    return tf.random.uniform(
        shape=(batch_size,),
        minval=int(action_spec.minimum),
        maxval=int(action_spec.maximum) + 1,
        dtype=tf.int32,
    )
