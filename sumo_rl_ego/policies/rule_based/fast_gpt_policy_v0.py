from ..base_policy import Policy
from ..registry import register_policy
from ...sumo_envs.highway_envs.highway_ego_v0 import ENV_PARAMS

'''

- if i am too close to the front vehicle, try to change lane (prefer left)
- elif i have enough space in front, accelerate
- elif i am too close but not critical, decelerate or same speed (ss)

'''

@register_policy("FastGPTPolicy-v0")
class FastGPTPolicy(Policy):
    def __init__(self):
        super().__init__()

        # --- thresholds ---
        self.free_front = 25.0
        self.close_front = 12.0
        self.critical_front = 7.0

        self.safe_front_lane = 15.0
        self.safe_back_lane = 10.0

    def predict(self, obs):
        ego_speed = obs[0] * ENV_PARAMS.max_speed

        lane_id = obs[1] * ENV_PARAMS.num_lanes

        lane_left_free = bool(obs[2])
        lane_right_free = bool(obs[3])
        
        d_front = obs[4] * ENV_PARAMS.max_distance 
        rel_speed_front = obs[5] * ENV_PARAMS.max_speed

        d_back = obs[6] * ENV_PARAMS.max_distance
        rel_speed_back = obs[7] * ENV_PARAMS.max_speed

        d_left_front = obs[8] * ENV_PARAMS.max_distance
        rel_speed_left_front = obs[9] * ENV_PARAMS.max_speed

        d_left_back = obs[10] * ENV_PARAMS.max_distance
        rel_speed_left_back = obs[11] * ENV_PARAMS.max_speed

        d_right_front = obs[12] * ENV_PARAMS.max_distance
        rel_speed_right_front = obs[13] * ENV_PARAMS.max_speed

        d_right_back = obs[14] * ENV_PARAMS.max_distance
        rel_speed_right_back = obs[15] * ENV_PARAMS.max_speed

        # --------------------------------------------------
        # Helper: lane safety
        # --------------------------------------------------
        def left_safe():
            return (
                lane_left_free and
                d_left_front > self.safe_front_lane and
                d_left_back > self.safe_back_lane
            )

        def right_safe():
            return (
                lane_right_free and
                d_right_front > self.safe_front_lane and
                d_right_back > self.safe_back_lane
            )

        # --------------------------------------------------
        # 1. CRITICAL → must react immediately
        # --------------------------------------------------
        if d_front < self.critical_front:
            if left_safe():
                return 3  # LCL
            if right_safe():
                return 4  # LCR
            return 2  # DEC

        # --------------------------------------------------
        # 2. CLOSE → try to overtake
        # --------------------------------------------------
        if d_front < self.close_front:
            if left_safe():
                return 3  # LCL (preferred)
            if right_safe():
                return 4  # LCR
            return 2  # DEC

        # --------------------------------------------------
        # 3. FREE → go fast
        # --------------------------------------------------
        if d_front > self.free_front:
            return 1  # ACC

        # --------------------------------------------------
        # 4. INTERMEDIATE → smooth behavior
        # --------------------------------------------------
        # not too close, not free → maintain or slight slow
        return 0  # SS