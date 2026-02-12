from .base_done import BaseDone

class CombinedDone(BaseDone):
    def __init__(self, *done_conditions):
        self.done_conditions = done_conditions

    def check(self, sim, ego, step_count):
        return any(cond.check(sim, ego, step_count)
                   for cond in self.done_conditions)