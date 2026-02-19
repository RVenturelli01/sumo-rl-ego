import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.observation.base_observation import BaseObservation
from gymnasium.spaces import MultiDiscrete

'''
Observation:
- lane availability (left, right)
- distance bin to leader (near, mid, far)
- distance bin to follower (near, mid, far)
- ego speed bin (0-5 m/s, 5-10 m/s, 10-15 m/s, ...)
'''


class MyObservation(BaseObservation):

    def __init__(self,
                near=30,
                far=50,
                lane_gap=10,
                speed_bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        ):

        self.near = near
        self.far = far
        self.lane_gap = lane_gap
        self.speed_bins = speed_bins    

        # numero bin velocità = len(bins)+1
        n_speed = len(speed_bins) + 1

        self.observation_space = MultiDiscrete([
            2,  # left free
            2,  # right free
            3,  # leader dist
            3,  # follower dist
            n_speed  # ego speed
        ])


    def build(self):

        # lane availability
        left_free  = self.lane_free("left", self.lane_gap)
        right_free = self.lane_free("right", self.lane_gap)

        # distanze
        d_front, d_back = self.front_back_distances()

        leader_bin   = self.distance_bin(d_front, self.near, self.far)
        follower_bin = self.distance_bin(d_back, self.near, self.far)

        ego_speed_bin = self.ego_speed_bin(self.speed_bins)

        return np.array([
            left_free,
            right_free,
            leader_bin,
            follower_bin,
            ego_speed_bin
        ], dtype=np.int32)
    
    
    # ==========================================================
    # Helpers
    # ==========================================================

    def front_back_distances(self):
        leader = self.sim.vehicle.getLeader(self.ego_id)
        follower = self.sim.vehicle.getFollower(self.ego_id)

        d_front = leader[1] if leader else float("inf")
        d_back  = follower[1] if follower else float("inf")

        return d_front, d_back


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
    

    def distance_bin(self, d, near, far):
        if d < near:
            return 0   # near
        elif d < far:
            return 1   # mid
        else:
            return 2   # far
    

    def ego_speed_bin(self, bins):
        v = self.sim.vehicle.getSpeed(self.ego_id)

        for i, b in enumerate(bins):
            if v < b:
                return i
        return len(bins)