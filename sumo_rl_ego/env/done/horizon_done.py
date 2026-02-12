from .base_done import BaseDone

class HorizonDone(BaseDone):
    def check(self, sim, ego, step_count):
        return step_count >= sim.simulation_end