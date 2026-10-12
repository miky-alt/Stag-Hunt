"""Learning-rate and entropy-coefficient schedules."""


def annealed_learning_rate(config, iteration):
    """Linear interpolation from learning_rate to learning_rate_end across iterations."""
    progress = min(iteration / max(config.num_iterations - 1, 1), 1.0)
    return config.learning_rate + progress * (config.learning_rate_end - config.learning_rate)


class EntropyScheduler:
    def __init__(self, agent, start_value=0.15, end_value=0.001, decay_steps=10000):
        self.agent = agent
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self, current_step):
        """Sets every trainer's entropy coefficient for this step and returns it."""
        if current_step >= self.decay_steps:
            new_entropy = self.end_value
        else:
            completion_fraction = current_step / self.decay_steps
            new_entropy = self.start_value - completion_fraction * (self.start_value - self.end_value)
        for trainer in self.agent.trainers:
            trainer.entropy_coef.assign(new_entropy)
        return new_entropy
