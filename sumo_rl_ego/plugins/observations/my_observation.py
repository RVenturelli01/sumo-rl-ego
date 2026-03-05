import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.sumo_gym_ego.observation.base import BaseObservationBuilder


class MyObservation(BaseObservationBuilder):

    def __init__(self,
                 max_speed=50.0,
                 max_distance=100.0,
                 max_rel_speed=30.0,
                 lane_gap=10):

        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_rel_speed = max_rel_speed
        self.lane_gap = lane_gap

        # 9 features continue
        self.observation_space = Box(
            low=np.array([
                0.0,        # ego speed
                0.0, 0.0,   # lane availability
                0.0, -1.0,  # front same lane (dist, rel speed)
                0.0, -1.0,  # front left
                0.0, -1.0   # front right
            ], dtype=np.float32),
            high=np.array([
                1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0,
                1.0, 1.0
            ], dtype=np.float32),
            dtype=np.float32
        )

    def build_obs(self):
        obs = []

        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)
        ego_speed_norm = ego_speed / self.max_speed
        obs.append(np.clip(ego_speed_norm, 0, 1))

        # lane availability
        left_free  = float(self.lane_free("left", self.lane_gap))
        right_free = float(self.lane_free("right", self.lane_gap))
        obs.extend([left_free, right_free])

        # FRONT SAME LANE
        front = self.sim.vehicle.getLeader(self.ego_id)
        if front is not None and front[0] in self.sim.vehicle.getIDList():
            front_id, d_front = front
            v_front = self.sim.vehicle.getSpeed(front_id)
            rel_speed_front = v_front - ego_speed
        else:
            d_front = self.max_distance
            rel_speed_front = self.max_rel_speed

        obs.append(self.norm_dist(d_front))
        obs.append(self.norm_rel_speed(rel_speed_front))

        # FRONT LEFT
        front_left = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
        if front_left:
            front_left_id, d_front_left = front_left[0]
            v_front_left = self.sim.vehicle.getSpeed(front_left_id)
            rel_speed_front_left = v_front_left - ego_speed
        else:
            d_front_left = self.max_distance
            rel_speed_front_left = self.max_rel_speed

        obs.append(self.norm_dist(d_front_left))
        obs.append(self.norm_rel_speed(rel_speed_front_left))

        # FRONT RIGHT
        front_right = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
        if front_right:
            front_right_id, d_front_right = front_right[0]
            v_front_right = self.sim.vehicle.getSpeed(front_right_id)
            rel_speed_front_right = v_front_right - ego_speed
        else:
            d_front_right = self.max_distance
            rel_speed_front_right = self.max_rel_speed

        obs.append(self.norm_dist(d_front_right))
        obs.append(self.norm_rel_speed(rel_speed_front_right))

        return np.array(obs, dtype=np.float32)


    def print_obs(self, obs):
        ego_speed = obs[0] * self.max_speed
        left_free = bool(obs[1])
        right_free = bool(obs[2])
        d_front = obs[3] * self.max_distance
        rel_speed_front = obs[4] * self.max_rel_speed
        d_front_left = obs[5] * self.max_distance
        rel_speed_front_left = obs[6] * self.max_rel_speed
        d_front_right = obs[7] * self.max_distance
        rel_speed_front_right = obs[8] * self.max_rel_speed

        print(f"Ego speed: {ego_speed:.2f} m/s")
        print(f"Lane availability Left, Right: | {left_free} | {right_free} |")
        print(f"Distances left, front, right: | {d_front_left:.2f} m | {d_front:.2f} m | {d_front_right:.2f} m |")
        print(f"Relative speed left, front, right: | {rel_speed_front_left:.2f} m/s | {rel_speed_front:.2f} m/s | {rel_speed_front_right:.2f} m/s |")
        
    

    # ------------------------
    # Utility functions
    # ------------------------
    def norm_dist(self, d):
        d = np.clip(d, 0, self.max_distance)
        return d / self.max_distance

    def norm_rel_speed(self, v):
        v = np.clip(v, -self.max_rel_speed, self.max_rel_speed)
        return v / self.max_rel_speed

    def lane_free(self, direction, safety_gap):

        # check if lane change in the given direction is possible (not at edge of road)
        idx = self.sim.vehicle.getLaneIndex(self.ego_id)
        lane_id = self.sim.vehicle.getLaneID(self.ego_id)
        edge = self.sim.lane.getEdgeID(lane_id)
        n = self.sim.edge.getLaneNumber(edge)

        if (direction == "left" and idx + 1 >= n) or \
        (direction == "right" and idx - 1 < 0):
            return 0
        
        # check if lane change in the given direction is safe (no vehicles within safety_gap)
        left_front = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
        left_back = self.sim.vehicle.getNeighbors(self.ego_id, 0b000)
        right_front = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
        right_back = self.sim.vehicle.getNeighbors(self.ego_id, 0b001)
        
        if direction == "left":
            neighbors = left_front + left_back
        else:            
            neighbors = right_front + right_back

        for _, gap in neighbors:
            if gap < safety_gap:
                return 0
        return 1