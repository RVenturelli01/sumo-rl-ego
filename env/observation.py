import numpy as np


class ObservationBuilder:
    def build(self, sim, ego):
        state = sim.get_vehicle_state(ego.id)

        obs = np.array([
            state["speed"],
            state["distance"]
        ], dtype=np.float32)

        return obs
