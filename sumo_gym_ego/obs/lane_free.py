import numpy as np
from gymnasium.spaces import Box
from sumo_gym_ego import BaseObservationBuilder


class LaneFreeObs(BaseObservationBuilder):

    def __init__(self):

        self.observation_space = Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64
        )

    def build_obs(self):

        # direction: +1 = left, -1 = right
        left_possible = self.sim.vehicle.couldChangeLane(self.ego_id, 1)
        right_possible = self.sim.vehicle.couldChangeLane(self.ego_id, -1)

        left_free = 1.0 if left_possible else 0.0
        right_free = 1.0 if right_possible else 0.0

        return np.array([left_free, right_free], dtype=np.float64)

    def print_obs(self, obs):

        left = "FREE" if obs[0] > 0.5 else "BLOCKED"
        right = "FREE" if obs[1] > 0.5 else "BLOCKED"

        print(f"{'left':>10} {'right':>10}")
        print(f"{left:>10} {right:>10}")