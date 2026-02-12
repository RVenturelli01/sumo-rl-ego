from .base_done import BaseDone

class OffRoadDone(BaseDone):
    def check(self, sim, ego, step_count):
        return sim.is_off_road()