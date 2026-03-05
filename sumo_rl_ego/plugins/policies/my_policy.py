import numpy as np
from sumo_rl_ego.infra.policy.base import BasePolicy

class MyPolicy(BasePolicy):
    """
    Discrete heuristic driving policy.

    Actions:
        0 = SS   (same speed)
        1 = ACC  (+acc_value)
        2 = DEC  (dec_value)
        3 = LCL
        4 = LCR
    """

    def __init__(
        self,
        max_speed=50.0,
        max_ttc=100.0,
        max_distance=100.0,
        lane_gap=8.0,        # minimum lateral safety distance
        acc_value=1.0,
        dec_value=-4.0,
        target_speed_ratio=0.8
    ):
        super().__init__()

        self.max_speed = max_speed
        self.max_ttc = max_ttc
        self.max_distance = max_distance
        self.lane_gap = lane_gap
        self.acc_value = acc_value
        self.dec_value = dec_value
        self.target_speed = max_speed * target_speed_ratio

    def predict(self, obs):
        # =========================
        # Decode observations
        # =========================
        ego_speed = obs[0] * self.max_speed

        d_front = obs[1] * self.max_distance
        d_left = obs[2] * self.max_distance
        d_right = obs[3] * self.max_distance

        ttc_front = obs[4] * self.max_ttc
        ttc_left = obs[5] * self.max_ttc
        ttc_right = obs[6] * self.max_ttc

        lane_index = obs[7]
        left_free = bool(obs[8])
        right_free = bool(obs[9])

        # =========================
        # Derived quantities
        # =========================

        # Relative speed estimation
        rel_speed = ego_speed - (d_front / max(ttc_front, 1e-3)) if ttc_front < self.max_ttc else 0.0

        # Braking distance (constant decel model)
        stopping_dist = (rel_speed ** 2) / (2 * abs(self.dec_value)) if rel_speed > 0 else 0.0

        need_brake = d_front < stopping_dist
        moderate_risk = d_front < 2.0 * stopping_dist

        # Lane safety checks
        left_safe = left_free and d_left > self.lane_gap and ttc_left > 3.0
        right_safe = right_free and d_right > self.lane_gap and ttc_right > 3.0

        # =========================
        # Decision logic
        # =========================

        # --- 1. Emergency avoidance ---
        if need_brake:
            if left_safe and (ttc_left >= ttc_right):
                return 3  # LCL
            if right_safe:
                return 4  # LCR
            return 2  # DEC fallback

        # --- 2. Preventive lane change (overtake logic) ---
        if moderate_risk:
            if left_safe and ttc_left > ttc_front:
                return 3
            if right_safe and ttc_right > ttc_front:
                return 4
            return 2  # slow down

        # --- 3. Opportunistic lane optimization ---
        # prefer lanes with more TTC even if not dangerous
        if left_safe and ttc_left > ttc_front + 2:
            return 3
        if right_safe and ttc_right > ttc_front + 2:
            return 4

        # --- 4. Speed control ---
        if ego_speed < self.target_speed:
            return 1  # ACC

        if ego_speed > self.max_speed * 0.95:
            return 0  # SS (avoid speeding)

        return 0