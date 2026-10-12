"""Maps the trainable state of an agent onto stable checkpoint keys."""


def checkpoint_items(agent, config, train_step_counter, heatmap_a, heatmap_b):
    """Returns the {name: trackable} mapping saved and restored by the checkpointer."""
    items = {
        'global_step': train_step_counter,
        'heatmap_globale_a': heatmap_a,
        'heatmap_globale_b': heatmap_b,
    }
    if agent.architecture == 'shared_shared':
        items.update({
            'actor_network': agent.actor,
            'critic_network': agent.critic,
            'optimizer': agent.optimizer,
        })
    elif agent.architecture == 'shared_actor_separate_critics':
        items['actor_network'] = agent.actor
        for index, trainer in enumerate(agent.trainers):
            items[f'critic_{index}'] = trainer.critic
            items[f'optimizer_{index}'] = trainer.optimizer
    else:
        for index, trainer in enumerate(agent.trainers):
            items[f'actor_{index}'] = trainer.actor
            items[f'critic_{index}'] = trainer.critic
            items[f'optimizer_{index}'] = trainer.optimizer

    if config.value_normalization:
        saved_normalizers = set()
        for index, trainer in enumerate(agent.trainers):
            normalizer = trainer.value_normalizer
            if normalizer is None or id(normalizer) in saved_normalizers:
                continue
            saved_normalizers.add(id(normalizer))
            items[f'value_norm_mean_{index}'] = normalizer.mean
            items[f'value_norm_variance_{index}'] = normalizer.variance
            items[f'value_norm_count_{index}'] = normalizer.count
    return items


def actor_checkpoint_items(actor_nets, train_step_counter):
    """Checkpoint keys for restoring only the actors (evaluation)."""
    items = {'global_step': train_step_counter}
    if len(actor_nets) == 1:
        items['actor_network'] = actor_nets[0]
    else:
        items.update({f'actor_{index}': actor for index, actor in enumerate(actor_nets)})
    return items
