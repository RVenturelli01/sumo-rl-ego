import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.sumo_gym_ego.observation.base import BaseObservationBuilder

'''
Observation schema:
- ego speed (normalized)
- distance front same lane (normalized)
- distance front left lane (normalized)
- distance front right lane (normalized)
- ttc front same lane (normalized)
- ttc front left (normalized)
- ttc front right (normalized)
- lane index (normalized)
- left lane free (binary)
- right lane free (binary)
'''


class SpeedDistTTCObs(BaseObservationBuilder):

    def __init__(self,
                 max_speed=50.0,
                 max_ttc=100,
                 max_distance=100.0,
                 lane_gap=5.0):

        self.max_speed = max_speed
        self.max_ttc = max_ttc
        self.max_distance = max_distance
        self.lane_gap = lane_gap

        # 9 features continue
        self.observation_space = Box(
            low=np.array([
                0.0,        # ego speed
                0.0,        # distance front same lane
                0.0,        # distance front left lane
                0.0,        # distance front right lane
                0.0,        # ttc front same lane
                0.0,        # ttc front left
                0.0,        # ttc front right
                0.0,        # lane index
                0.0,        # left lane free
                0.0,        # right lane free
            ], dtype=np.float32),
            high=np.array([
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ], dtype=np.float32),
            dtype=np.float32
        )


    def set_context(self, ctx):
        super().set_context(ctx)
        self.ctx.inject("obs_max_speed", self.max_speed)
        self.ctx.inject("obs_max_ttc", self.max_ttc)
        self.ctx.inject("obs_max_distance", self.max_distance)


    def build_obs(self):

        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)
        ego_speed_norm = np.clip(ego_speed, 0, self.max_speed) / self.max_speed


        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)
        num_lanes = self.sim.edge.getLaneNumber(
            self.sim.lane.getEdgeID(self.sim.vehicle.getLaneID(self.ego_id))
        )
        
        # distance front same lane
        front = self.sim.vehicle.getLeader(self.ego_id)
        if front is not None and front[0] in self.sim.vehicle.getIDList():
            _, d_front = front
        else:
            d_front = float('inf') 
        d_front_norm = np.clip(d_front, 0, self.max_distance) / self.max_distance


        # distance front same lane
        front = self.sim.vehicle.getLeader(self.ego_id)
        if front is not None and front[0] in self.sim.vehicle.getIDList():
            _, d_front = front
        else:
            d_front = float('inf') 
        d_front_norm = np.clip(d_front, 0, self.max_distance) / self.max_distance


        # distance front left lane
        if lane_index < num_lanes - 1:  # left exists
            front_left = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
            if front_left:
                _, d_front_left = front_left[0]
            else:
                d_front_left = float('inf') 
        else:
            d_front_left = float('inf')
        d_front_left_norm = np.clip(d_front_left, 0, self.max_distance) / self.max_distance


        # distance front right lane
        if lane_index > 0:  # right exists
            front_right = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
            if front_right:
                _, d_front_right = front_right[0]
            else:
                d_front_right = float('inf') 
        else:
            d_front_right = float('inf')
        d_front_right_norm = np.clip(d_front_right, 0, self.max_distance) / self.max_distance


        # ttc front same lane
        front = self.sim.vehicle.getLeader(self.ego_id)
        if front is not None and front[0] in self.sim.vehicle.getIDList():
            front_id, d_front = front
            v_front = self.sim.vehicle.getSpeed(front_id)
            rel_speed_front = ego_speed - v_front
            ttc_front = d_front / rel_speed_front if rel_speed_front > 0 else float('inf')
        else:
            ttc_front = float('inf')  
        ttc_front_norm = np.clip(ttc_front, 0, self.max_ttc) / self.max_ttc
        

        # ttc front left
        if lane_index < num_lanes - 1:  # left exists
            front_left = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
            if front_left:
                front_left_id, d_front_left = front_left[0]
                v_front_left = self.sim.vehicle.getSpeed(front_left_id)
                rel_speed_front_left = ego_speed - v_front_left
                ttc_front_left = d_front_left / rel_speed_front_left if rel_speed_front_left > 0 else float('inf')
            else:
                ttc_front_left = float('inf')   
        else:
            ttc_front_left = 0.0
        ttc_front_left_norm = np.clip(ttc_front_left, 0, self.max_ttc) / self.max_ttc


        # ttc front right
        if lane_index > 0:  # right exists
            front_right = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
            if front_right:
                front_right_id, d_front_right = front_right[0]
                v_front_right = self.sim.vehicle.getSpeed(front_right_id)
                rel_speed_front_right = ego_speed - v_front_right
                ttc_front_right = d_front_right / rel_speed_front_right if rel_speed_front_right > 0 else float('inf')
            else:
                ttc_front_right = float('inf')
        else:
            ttc_front_right = 0.0
        ttc_front_right_norm = np.clip(ttc_front_right, 0, self.max_ttc) / self.max_ttc

        # lane index normalized
        lane_index_norm = lane_index / max(num_lanes - 1, 1)

        # lane free
        left_free  = float(self.lane_free("left", self.lane_gap))
        right_free = float(self.lane_free("right", self.lane_gap))


        return np.array([
                ego_speed_norm,
                d_front_norm,
                d_front_left_norm,
                d_front_right_norm,
                ttc_front_norm,
                ttc_front_left_norm,
                ttc_front_right_norm,
                lane_index_norm,
                left_free,
                right_free
             ], dtype=np.float32)


    def com(self, obs):
        ego_speed = obs[0] * self.max_speed
        distance_front = obs[1] * self.max_distance
        distance_front_left = obs[2] * self.max_distance
        distance_front_right = obs[3] * self.max_distance
        ttc_front = obs[4] * self.max_ttc   
        ttc_front_left = obs[5] * self.max_ttc
        ttc_front_right = obs[6] * self.max_ttc
        lane_index = obs[7] * (self.sim.edge.getLaneNumber(
            self.sim.lane.getEdgeID(self.sim.vehicle.getLaneID(self.ego_id))
        ) - 1)
        left_free = bool(obs[8])
        right_free = bool(obs[9])

        print(f"Ego speed: {ego_speed:.1f} m/s")
        print(f"Distance front: | {distance_front:.1f} m | {distance_front_left:.1f} m | {distance_front_right:.1f} m |")
        print(f"TTC front: | {ttc_front_left:.1f} s | {ttc_front:.1f} s | {ttc_front_right:.1f} s |")
        print(f"Lane index: {lane_index:.0f}")
        print(f"Free lane: | {left_free} | {right_free} |")


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