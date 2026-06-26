import numpy as np
from gymnasium.spaces import Box
from sumo_gym_ego import BaseObservationBuilder


class LaneFreeObs(BaseObservationBuilder):

    def __init__(self, check_distance=20.0):

        self.check_distance = check_distance

        self.observation_space = Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float64
        )

    def build_obs(self):

        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)
        road_id = self.sim.vehicle.getRoadID(self.ego_id)
        num_lanes = self.sim.edge.getLaneNumber(road_id)

        left_free = 1.0
        right_free = 1.0

        ego_pos = self.sim.vehicle.getLanePosition(self.ego_id)

        # check left lane
        if lane_index < num_lanes - 1:
            left_lane = lane_index + 1
            vehs = self.sim.lane.getLastStepVehicleIDs(f"{road_id}_{left_lane}")

            for v in vehs:
                pos = self.sim.vehicle.getLanePosition(v)
                if abs(pos - ego_pos) < self.check_distance:
                    left_free = 0.0
                    break
        else:
            left_free = 0.0  # lane does not exist

        # check right lane
        if lane_index > 0:
            right_lane = lane_index - 1
            vehs = self.sim.lane.getLastStepVehicleIDs(f"{road_id}_{right_lane}")

            for v in vehs:
                pos = self.sim.vehicle.getLanePosition(v)
                if abs(pos - ego_pos) < self.check_distance:
                    right_free = 0.0
                    break
        else:
            right_free = 0.0  # lane does not exist

        return np.array([left_free, right_free], dtype=np.float64)


    def format_obs(self, obs) -> str:
        left = "FREE" if obs[0] > 0.5 else "BLOCKED"
        right = "FREE" if obs[1] > 0.5 else "BLOCKED"
        return f"{'left':>10} {'right':>10}\n{left:>10} {right:>10}"