import numpy as np
from gymnasium.spaces import Box
from sumo_gym_ego import BaseObservationBuilder


class EgoSpeedObs(BaseObservationBuilder):

    def __init__(self, max_speed=50):

        self.max_speed = max_speed

        self.observation_space = Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64
        )

    def build_obs(self):

        speed = self.sim.vehicle.getSpeed(self.ego_id)
        speed_norm = np.clip(speed / self.max_speed, 0, 1)

        return np.array([speed_norm], dtype=np.float64)

    def format_obs(self, obs) -> str:
        speed = float(obs[0]) * self.max_speed
        return f"ego_speed : {speed:.3f} m/s"


class EgoLaneObs(BaseObservationBuilder):

    def __init__(self):

        self.observation_space = Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float64
        )

    def build_obs(self):

        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)

        lane_id = self.sim.vehicle.getLaneID(self.ego_id)
        edge = self.sim.lane.getEdgeID(lane_id)
        num_lanes = self.sim.edge.getLaneNumber(edge)

        lane_norm = lane_index / max(num_lanes - 1, 1)

        return np.array([lane_norm], dtype=np.float64)
    

    def format_obs(self, obs) -> str:
        lane_id = self.sim.vehicle.getLaneID(self.ego_id)
        edge = self.sim.lane.getEdgeID(lane_id)
        num_lanes = self.sim.edge.getLaneNumber(edge)
        lane_index = float(obs[0]) * max(num_lanes - 1, 1)
        return f"ego_lane : {lane_index:.3f}"