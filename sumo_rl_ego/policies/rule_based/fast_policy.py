from ..base_policy import Policy
from ..registry import register_policy
from sumo_gym_ego.sumo_envs.highway_envs.highway_ego_v0 import ENV_PARAMS

'''

- if i am too close to the front vehicle, try to change lane (prefer left)
- elif i have enough space in front, accelerate
- elif i am too close but not critical, decelerate or same speed (ss)

'''

@register_policy("FastPolicy-v0")
class FastPolicy(Policy):
    def __init__(self):
        super().__init__()

        # --- thresholds ---
        self.free_front = 50.0
        self.close_front = 30.0

        # braking model
        self.max_decel = ENV_PARAMS.dec_value

    # --------------------------------------------------
    # Braking feasibility
    # --------------------------------------------------
    def dist_required(self, d, rel_speed):
        if rel_speed <= 0:
            return 0.0

        d_required = (rel_speed ** 2) / (2 * abs(self.max_decel))  
        return max(d_required, 10)


    def predict(self, obs):
        ego_speed = obs[0] * ENV_PARAMS.max_speed

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
        # Safety check (current lane)
        # --------------------------------------------------
        safe_brake_dist = self.dist_required(d_front, rel_speed_front)
        safe_brake_dist_left = self.dist_required(d_left_front, rel_speed_left_front)
        safe_brake_dist_right = self.dist_required(d_right_front, rel_speed_right_front)

        # --------------------------------------------------
        # Helper: lane safety (WITH BRAKING MODEL)
        # --------------------------------------------------
        def left_safe():
            return (
                lane_left_free and
                d_left_front > safe_brake_dist_left
            )

        def right_safe():
            return (
                lane_right_free and
                d_right_front > safe_brake_dist_right
            )

        # --------------------------------------------------
        # 1. CRITICAL → cannot brake → MUST react
        # --------------------------------------------------
        if d_front < safe_brake_dist:
            if left_safe():
                return 3  # LCL
            if right_safe():
                return 4  # LCR
            return 2  # DEC and pray

        # --------------------------------------------------
        # 2. CLOSE → approaching limit
        # --------------------------------------------------
        if d_front < safe_brake_dist*3:
            if left_safe() and d_left_front > safe_brake_dist_left*3:
                return 3
            if right_safe() and d_right_front > safe_brake_dist_right*3:
                return 4
            return 2

        # --------------------------------------------------
        # 3. FREE → go fast
        # --------------------------------------------------
        if d_front < safe_brake_dist*4:
            return 0  # SS

        return 1  # ACC