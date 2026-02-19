import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.observation.base_observation import BaseObservation
from gymnasium.spaces import MultiDiscrete

'''
Observation:
- distance bin to leader (near, mid, far)
- distance bin to follower (near, mid, far)
- ego speed bin (0-5 m/s, 5-10 m/s, 10-15 m/s, 15-20 m/s, >20 m/s)
'''


class MyObservation(BaseObservation):

    def __init__(self,
                 ego_speed_bins=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                 distance_bins=[10, 20, 30, 40],
                 ego_speed_veh_bins=[5, 10, 15, 20, 25, 30],
                 ):
        
        self.distance_bins = distance_bins
        self.ego_speed_veh_bins = ego_speed_veh_bins    
        self.ego_speed_bins = ego_speed_bins    

        # numero bin = len(bins) + 1
        n_distance = len(distance_bins) + 1
        n_speed_veh = len(ego_speed_veh_bins) + 1
        n_speed = len(ego_speed_bins) + 1

        neighbor_state = n_distance + n_speed_veh

        self.observation_space = MultiDiscrete(
            [n_speed] +             # ego speed
            [3] +                   # lane
            ([n_distance, n_speed_veh] * 6)
        )

    def build(self):
        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)
        ego_speed_bin = self.bin_value(ego_speed, self.ego_speed_bins)

        ego_lane = self.sim.vehicle.getLaneIndex(self.ego_id)

        # -----------------------
        # LATERAL
        # -----------------------
        left_front = self.closest(self.sim.vehicle.getNeighbors(self.ego_id, 0b010))
        left_back  = self.closest(self.sim.vehicle.getNeighbors(self.ego_id, 0b000))
        right_front = self.closest(self.sim.vehicle.getNeighbors(self.ego_id, 0b011))
        right_back  = self.closest(self.sim.vehicle.getNeighbors(self.ego_id, 0b001))

        # -----------------------
        # SAME LANE
        # -----------------------
        front = self.normalize_single(self.sim.vehicle.getLeader(self.ego_id))
        back  = self.normalize_single(self.sim.vehicle.getFollower(self.ego_id))

        neighbors = [left_front, left_back, right_front, right_back, front, back]
        obs = np.array([ego_speed_bin, ego_lane], dtype=np.int32)

        for i, neighbor in enumerate(neighbors):
            if neighbor[0] is not None and neighbor[0] in self.sim.vehicle.getIDList():
                d = neighbor[1]
                v = self.sim.vehicle.getSpeed(neighbor[0])
            else:
                d = float("inf")
                v = 0

            distance_bin = self.bin_value(d, self.distance_bins)
            speed_bin = self.bin_value(v, self.ego_speed_veh_bins)

            obs = np.append(obs, [distance_bin, speed_bin])
    
        return obs
    

    def bin_value(self, value, bins):
        for i, b in enumerate(bins):
            if value < b:
                return i
        return len(bins)
    

    def closest(self, neigh):
        if not neigh:
            return None, None
        return min(neigh, key=lambda x: abs(x[1]))


    def normalize_single(self, x):
        return x if x else (None, None)
