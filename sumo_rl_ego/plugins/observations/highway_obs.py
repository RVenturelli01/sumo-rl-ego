import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.sumo_gym_ego.observation.base import BaseObservationBuilder

'''
ego_speed
lane_index

same_front  : distance, rel_speed, ttc
same_back   : distance, rel_speed, ttc

left_front  : distance, rel_speed, ttc
left_back   : distance, rel_speed, ttc

right_front : distance, rel_speed, ttc
right_back  : distance, rel_speed, ttc
'''

class HighwayObs(BaseObservationBuilder):

    def __init__(
        self,
        max_speed=50.0,
        max_distance=100.0,
        max_ttc=20.0,
    ):

        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_ttc = max_ttc

        n_features = 20

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(n_features,),
            dtype=np.float32
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def norm_speed(self, v):
        return np.clip(v, 0, self.max_speed) / self.max_speed

    def norm_distance(self, d):
        return np.clip(d, 0, self.max_distance) / self.max_distance

    def norm_ttc(self, ttc):
        return np.clip(ttc, 0, self.max_ttc) / self.max_ttc


    # ============================================================
    # TTC
    # ============================================================

    def compute_ttc(self, distance, rel_speed):

        if rel_speed <= 0:
            return float("inf")

        return distance / rel_speed

    # ============================================================
    # VEHICLE FEATURES
    # ============================================================

    def vehicle_features(self, neigh, ego_speed):

        if neigh is None:
            return [1.0, 0.5, 1.0]

        veh_id, distance = neigh

        if veh_id not in self.sim.vehicle.getIDList():
            return [1.0, 0.5, 1.0]

        v = self.sim.vehicle.getSpeed(veh_id)

        rel_speed = ego_speed - v
        ttc = self.compute_ttc(distance, rel_speed)

        return [
            self.norm_distance(distance),
            self.norm_speed(rel_speed),
            self.norm_ttc(ttc)
        ]

    # ============================================================
    # NEIGHBOR EXTRACTION LAYER
    # ============================================================

    def extract_neighbors(self):

        neighbors = {}

        # same lane
        neighbors["same_front"] = self.sim.vehicle.getLeader(self.ego_id)

        neighbors["same_back"] = self.sim.vehicle.getFollower(self.ego_id)

        # left lane
        neigh = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
        neighbors["left_front"] = neigh[0] if neigh else None

        neigh = self.sim.vehicle.getNeighbors(self.ego_id, 0b000)
        neighbors["left_back"] = neigh[0] if neigh else None

        # right lane
        neigh = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
        neighbors["right_front"] = neigh[0] if neigh else None

        neigh = self.sim.vehicle.getNeighbors(self.ego_id, 0b001)
        neighbors["right_back"] = neigh[0] if neigh else None

        return neighbors

    # ============================================================
    # OBSERVATION
    # ============================================================

    def build_obs(self):

        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)
        ego_speed_norm = self.norm_speed(ego_speed)

        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)

        lane_id = self.sim.vehicle.getLaneID(self.ego_id)
        edge = self.sim.lane.getEdgeID(lane_id)
        num_lanes = self.sim.edge.getLaneNumber(edge)

        lane_norm = lane_index / max(num_lanes - 1, 1)

        obs = [ego_speed_norm, lane_norm]

        neigh = self.extract_neighbors()

        order = [
            "same_front",
            "same_back",
            "left_front",
            "left_back",
            "right_front",
            "right_back",
        ]

        for key in order:
            obs.extend(self.vehicle_features(neigh[key], ego_speed))

        return np.array(obs, dtype=np.float32)

    # ============================================================
    # DEBUG PRINT
    # ============================================================

    def print_obs(self, obs):

        ego_speed = obs[0] * self.max_speed

        lane_index = obs[1] * (
            self.sim.edge.getLaneNumber(
                self.sim.lane.getEdgeID(self.sim.vehicle.getLaneID(self.ego_id))
            ) - 1
        )

        def decode_vehicle(i):
            d = obs[i] * self.max_distance
            rel_v = (obs[i+1] * 2 - 1) * self.max_speed
            ttc = obs[i+2] * self.max_ttc
            return d, rel_v, ttc

        # ordine delle osservazioni
        idx = {
            "same_front": 2,
            "same_back": 5,
            "left_front": 8,
            "left_back": 11,
            "right_front": 14,
            "right_back": 17,
        }

        lf = decode_vehicle(idx["left_front"])
        sf = decode_vehicle(idx["same_front"])
        rf = decode_vehicle(idx["right_front"])

        lb = decode_vehicle(idx["left_back"])
        sb = decode_vehicle(idx["same_back"])
        rb = decode_vehicle(idx["right_back"])

        print("\n========== OBS ==========")
        print(f"Ego speed : {ego_speed:.2f} m/s")
        print(f"Lane index: {lane_index:.1f}\n")

        print(f"{'':<10}{'LEFT':>12}{'SAME':>12}{'RIGHT':>12}")
        print("-" * 46)

        print(f"{'dist_f':<10}{lf[0]:>12.1f}{sf[0]:>12.1f}{rf[0]:>12.1f}")
        print(f"{'rel_v_f':<10}{lf[1]:>12.1f}{sf[1]:>12.1f}{rf[1]:>12.1f}")
        print(f"{'ttc_f':<10}{lf[2]:>12.1f}{sf[2]:>12.1f}{rf[2]:>12.1f}")

        print("-" * 46)

        print(f"{'dist_b':<10}{lb[0]:>12.1f}{sb[0]:>12.1f}{rb[0]:>12.1f}")
        print(f"{'rel_v_b':<10}{lb[1]:>12.1f}{sb[1]:>12.1f}{rb[1]:>12.1f}")
        print(f"{'ttc_b':<10}{lb[2]:>12.1f}{sb[2]:>12.1f}{rb[2]:>12.1f}")

        print("=" * 46 + "\n")