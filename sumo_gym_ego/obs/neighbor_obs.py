import numpy as np
from gymnasium.spaces import Box
from sumo_gym_ego import BaseObservationBuilder


class NeighborObs(BaseObservationBuilder):

    AVAILABLE_NEIGHBORS = {
        "same_front":  (None,  1),
        "same_back":   (None, -1),

        "left_front":  (0b010,  1),
        "left_back":   (0b000, -1),

        "right_front": (0b011,  1),
        "right_back":  (0b001, -1),
    }

    AVAILABLE_FEATURES = ["distance", "rel_speed", "ttc"]

    def __init__(
        self,
        neighbors=None,
        features=None,
        max_speed=50,
        max_distance=200,
        max_ttc=20,
    ):

        self.neighbors = neighbors
        self.features = features

        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_ttc = max_ttc

        n_features = len(self.neighbors) * len(self.features)

        self.observation_space = Box(
            low=-1,
            high=1,
            shape=(n_features,),
            dtype=np.float64,
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    def norm_distance(self, d):
        return np.clip(d, -self.max_distance, self.max_distance) / self.max_distance

    def norm_rel_speed(self, v):
        return np.clip(v, -self.max_speed, self.max_speed) / self.max_speed

    def norm_ttc(self, t):
        return np.clip(t, 0, self.max_ttc) / self.max_ttc

    # ============================================================
    # TTC
    # ============================================================

    def compute_ttc(self, distance, rel_speed):

        if rel_speed <= 0:
            return self.max_ttc

        return min(distance / rel_speed, self.max_ttc)

    # ============================================================
    # NEIGHBOR EXTRACTION
    # ============================================================

    def get_neighbor(self, name):

        mask, sign = self.AVAILABLE_NEIGHBORS[name]

        if name == "same_front":
            neigh = self.sim.vehicle.getLeader(self.ego_id)

        elif name == "same_back":
            neigh = self.sim.vehicle.getFollower(self.ego_id)

        else:
            neigh = self.sim.vehicle.getNeighbors(self.ego_id, mask)
            neigh = neigh[0] if neigh else None

        return neigh, sign

    # ============================================================
    # FEATURE COMPUTATION
    # ============================================================

    def compute_feature(self, feature, neigh, ego_speed, sign):

        if neigh is None:
            if feature == "distance":
                return sign
            if feature == "rel_speed":
                return -sign
            if feature == "ttc":
                return 1

        veh_id, distance = neigh

        if veh_id not in self.sim.vehicle.getIDList():
            if feature == "distance":
                return sign
            if feature == "rel_speed":
                return -sign
            if feature == "ttc":
                return 1

        v = self.sim.vehicle.getSpeed(veh_id)

        rel_speed = ego_speed - v

        if feature == "rel_speed":
            return self.norm_rel_speed(rel_speed)
        

        ego_pos = self.sim.vehicle.getLanePosition(self.ego_id)
        veh_pos = self.sim.vehicle.getLanePosition(veh_id)
        distance = veh_pos - ego_pos

        if feature == "distance":
            return self.norm_distance(distance)

        if feature == "ttc":
            ttc = self.compute_ttc(abs(distance), sign*rel_speed)
            return self.norm_ttc(ttc)

        raise ValueError(feature)

    # ============================================================
    # OBSERVATION
    # ============================================================

    def build_obs(self):

        ego_speed = self.sim.vehicle.getSpeed(self.ego_id)

        obs = []

        for n in self.neighbors:

            neigh, sign = self.get_neighbor(n)

            for f in self.features:
                obs.append(
                    self.compute_feature(f, neigh, ego_speed, sign)
                )

        return np.array(obs, dtype=np.float64)
    

    def print_obs(self, obs):

        rows = ["front", "back"]
        cols = ["left", "same", "right"]

        idx = 0

        # costruisco tabelle per feature
        tables = {
            f: {(r, c): "-" for r in rows for c in cols}
            for f in self.features
        }

        for n in self.neighbors:

            side, pos = n.split("_")

            for f in self.features:

                val = float(obs[idx])

                if f == "distance":
                    val = val * self.max_distance
                    val = f"{val:.1f}"

                elif f == "rel_speed":
                    val = val * self.max_speed
                    val = f"{val:.1f}"

                elif f == "ttc":
                    val = val * self.max_ttc
                    val = f"{val:.1f}"

                tables[f][(pos, side)] = val

                idx += 1

        # stampa
        for f in self.features:

            print(f"\n[{f}]")
            print(f"{'':8s}{'left':>12}{'same':>12}{'right':>12}")

            for r in rows:
                print(
                    f"{r:8s}"
                    f"{tables[f][(r,'left')]:>12}"
                    f"{tables[f][(r,'same')]:>12}"
                    f"{tables[f][(r,'right')]:>12}"
                )