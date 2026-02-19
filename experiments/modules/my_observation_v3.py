import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.observation.base_observation import BaseObservation
from gymnasium.spaces import MultiDiscrete

'''
Observation:
'''


class MyObservation(BaseObservation):

    def __init__(self,
                 ego_speed_bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                 distance_bins=[10, 20, 30, 40],
                 rel_ego_veh_speed_bins=[-10, -5, -2, 2, 5, 10],
                 lane_gap=10,
                 ):
        
        self.distance_bins = distance_bins
        self.rel_ego_veh_speed_bins = rel_ego_veh_speed_bins    
        self.ego_speed_bins = ego_speed_bins    
        self.lane_gap = lane_gap

        # numero bin = len(bins) + 1
        n_distance = len(distance_bins) + 1
        n_speed_veh = len(rel_ego_veh_speed_bins) + 1
        n_speed_ego = len(ego_speed_bins) + 1

        self.observation_space = MultiDiscrete(
            [n_speed_ego] +               # ego speed
            [2, 2] +                      # left lane free, right lane free
            [n_distance, n_speed_veh] +   # front same lane
            [n_distance, n_speed_veh] +   # front left
            [n_distance, n_speed_veh]     # front right
        )

    def build(self):
        obs = []

        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)
        ego_speed_bin = self.bin_value(ego_speed, self.ego_speed_bins)
        obs.append(ego_speed_bin)


        # lane availability
        left_free  = self.lane_free("left", self.lane_gap)
        right_free = self.lane_free("right", self.lane_gap)
        obs.append(left_free)
        obs.append(right_free)

        # front same lane
        front = self.sim.vehicle.getLeader(self.ego_id)
        if front is not None and front[0] in self.sim.vehicle.getIDList():
            front_id, d_front = front
            v_front = self.sim.vehicle.getSpeed(front_id)
            rel_speed_front = v_front - ego_speed
        else:
            d_front = float("inf")
            rel_speed_front = float("inf")

        obs.append(self.bin_value(d_front, self.distance_bins))
        obs.append(self.bin_value(rel_speed_front, self.rel_ego_veh_speed_bins))
        
        # front left
        front_left = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
        if front_left:
            front_left_id, d_front_left = front_left[0]  
            v_front_left = self.sim.vehicle.getSpeed(front_left_id)
            rel_speed_front_left = v_front_left - ego_speed 
        else:
            d_front_left = float("inf")
            rel_speed_front_left = float("inf")

        obs.append(self.bin_value(d_front_left, self.distance_bins))    
        obs.append(self.bin_value(rel_speed_front_left, self.rel_ego_veh_speed_bins))


        # front right
        front_right = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
        if front_right:         
            front_right_id, d_front_right = front_right[0]
            v_front_right = self.sim.vehicle.getSpeed(front_right_id)
            rel_speed_front_right = v_front_right - ego_speed
        else:
            d_front_right = float("inf")
            rel_speed_front_right = float("inf")

        obs.append(self.bin_value(d_front_right, self.distance_bins))
        obs.append(self.bin_value(rel_speed_front_right, self.rel_ego_veh_speed_bins))

        return np.array(obs, dtype=np.int32)
    

    def print_obs(self, obs):
        ego_speed_bin = obs[0]
        left_free = obs[1]
        right_free = obs[2]
        front_d_bin = obs[3]
        front_rel_speed_bin = obs[4]
        front_left_d_bin = obs[5]
        front_left_rel_speed_bin = obs[6]
        front_right_d_bin = obs[7]
        front_right_rel_speed_bin = obs[8]

        print(f"EGO speed bin: {ego_speed_bin}")
        print(f"Left lane free: {left_free} | Right lane free: {right_free}")
        print(f"Front same lane - distance bin: {front_d_bin} | relative speed bin: {front_rel_speed_bin}")
        print(f"Front left - distance bin: {front_left_d_bin} | relative speed bin: {front_left_rel_speed_bin}")
        print(f"Front right - distance bin: {front_right_d_bin} | relative speed bin: {front_right_rel_speed_bin}")





    def bin_value(self, value, bins):
        for i, b in enumerate(bins):
            if value < b:
                return i
        return len(bins)
    

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