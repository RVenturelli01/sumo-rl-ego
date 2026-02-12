class DoneCondition:
    def __init__(self, max_steps=1000):
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        self.steps = 0

    def check(self, sim, ego):
        self.steps += 1

        if sim.has_collision():
            return True

        if self.steps >= self.max_steps:
            return True

        return False
