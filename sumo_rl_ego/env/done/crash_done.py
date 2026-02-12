from .base_done import BaseDone

class CrashDone(BaseDone):
    def check(self, sim, ego, step_count):
        return sim.has_crashed(ego.id)