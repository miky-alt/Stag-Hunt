class EntropyScheduler:
    def __init__(self, agent, start_value=0.15, end_value=0.001, decay_steps=10000):
        self.agent = agent
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps

    def step(self, current_step):
        if current_step >= self.decay_steps:
            new_entropy = self.end_value
        else:
            completion_fraction = current_step / self.decay_steps
            new_entropy = self.start_value - completion_fraction * (
                self.start_value - self.end_value
            )

        for trainer in self.agent.trainers:
            trainer.entropy_coef.assign(new_entropy)
        return new_entropy
