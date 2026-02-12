import numpy as np
from gymnasium.spaces import Box
from .base_observation import BaseObservation


class KinLaneDistObservation(BaseObservation):

    def __init__(self):
        # [x, y, angle, speed, lane_pos, lane_pos_lat, lane_index, distance]
        self.observation_space = Box(
            low=np.array([
                -1e5,   # x
                -1e5,   # y
                -360.0, # angle
                0.0,    # speed
                0.0,    # lane position
                -10.0,  # lateral lane position
                0.0,    # lane index
                0.0     # distance
            ], dtype=np.float32),

            high=np.array([
                1e5,    # x
                1e5,    # y
                360.0,  # angle
                100.0,  # speed
                1e4,    # lane position
                10.0,   # lateral lane position
                10.0,   # lane index
                1e4     # distance
            ], dtype=np.float32),

            dtype=np.float32
        )

    def build(self, sim, ego):

        x, y = sim.get_position(ego.id)
        angle = sim.get_angle(ego.id)
        speed = sim.get_speed(ego.id)
        lane_pos = sim.get_lane_position(ego.id)
        lane_pos_lat = sim.get_lane_position_lat(ego.id)
        lane_index = sim.get_lane_index(ego.id)
        distance = sim.get_distance(ego.id)

        return np.array([
            x,
            y,
            angle,
            speed,
            lane_pos,
            lane_pos_lat,
            lane_index,
            distance
        ], dtype=np.float32)
