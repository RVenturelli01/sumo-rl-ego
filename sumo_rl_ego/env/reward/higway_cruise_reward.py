from sumo_rl_ego.env.reward.base_reward import BaseReward

"""
DrivingRLReward

Per-step reward for highway RL.

Rewards:
- Speed close to target_speed.
- Smooth driving (low acceleration/deceleration).
- Lane stability.

Penalizes:
- Strong accelerations and harsh braking.
- Unnecessary lane changes.
- Off-road events.
- Collisions (strong penalty).

Expected behavior: stable speed near target, smooth control,
lane changes only when needed, no unsafe events.
"""

class HighwayCruiseReward(BaseReward):

    def __init__(
        self,
        target_speed=25.0,        # m/s
        max_speed=30.0,
        w_speed=1.0,
        w_accel=-0.02,
        w_comfort_decel=-0.05,
        w_harsh_decel=-0.2,
        w_lane_change=-0.05,
        w_offroad=-5.0,
        w_crash=-10.0,
        hsd_threshold=-3.0       # m/s^2  (Harsh Sudden Deceleration threshold)
    ):
        self.target_speed = target_speed
        self.max_speed = max_speed

        self.w_speed = w_speed
        self.w_accel = w_accel
        self.w_comfort_decel = w_comfort_decel
        self.w_harsh_decel = w_harsh_decel
        self.w_lane_change = w_lane_change
        self.w_offroad = w_offroad
        self.w_crash = w_crash

        self.HSD_THRESHOLD = hsd_threshold


        # memorizziamo stato precedente per reward per-step
        self.prev_speed = None


    def compute(self, sim, ego):

        reward = 0.0

        speed = sim.get_speed(ego.id)

        # =========================
        # 1) SPEED TRACKING TERM
        # =========================
        # reward massima vicino a target_speed
        speed_error = abs(speed - self.target_speed)
        speed_reward = 1 - (speed_error / self.max_speed)
        reward += self.w_speed * speed_reward


        # =========================
        # 2) SMOOTHNESS TERM
        # =========================
        if self.prev_speed is not None:
            accel = (speed - self.prev_speed) / sim.time_step

            if accel > 0:
                reward += self.w_accel * accel
            elif accel < 0:
                if accel < self.HSD_THRESHOLD:
                    reward += self.w_harsh_decel * abs(accel)
                else:
                    reward += self.w_comfort_decel * abs(accel)

        self.prev_speed = speed


        # =========================
        # 3) LANE CHANGE PENALTY
        # =========================
        if hasattr(sim, "last_lane_change") and sim.last_lane_change:
            reward += self.w_lane_change

        return reward


    def compute_terminal(
        self,
        has_collided: bool,
        has_teleported: bool,
        is_off_road: bool,
        route_completed: bool,
        ego_removed_unknown: bool,
        truncated_due_to_timeout: bool,
    ):

        reward = 0.0

        if has_collided:
            reward = self.w_crash

        elif is_off_road:
            reward = self.w_offroad

        return reward