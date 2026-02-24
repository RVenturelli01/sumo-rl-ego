import numpy as np
from gymnasium.spaces import Box
from sumo_rl_ego.ego.base import BaseEgoController


class HighwayContinuousEgo(BaseEgoController):
    """
    Continuous ego controller:
    action = [acceleration, lane_command]
    """

    def __init__(
        self,
        max_acc=2.0,        # m/s^2
        max_dec=-4.0,       # m/s^2
        lc_duration=0,
        lane_threshold=0.5  # threshold for lane change
    ):
        self.max_acc = max_acc
        self.max_dec = max_dec
        self.lc_duration = lc_duration
        self.lane_threshold = lane_threshold

        # action: [a_long, lane_signal]
        self.action_space = Box(
            low=np.array([-1, -1.0], dtype=np.float32),
            high=np.array([1, 1.0], dtype=np.float32),
        )

    def apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)

        time_step = self.sim.config.time_step
        speed = self.sim.vehicle.getSpeed(self.ego_id)

        # -------- NORMALIZED -> PHYSICAL --------
        accel = np.clip(action[0], -1, 1)*(self.max_acc+self.max_dec)/2 + (self.max_acc-self.max_dec)/2
        new_speed = max(0.0, speed + accel * time_step)
        self.sim.vehicle.setSpeed(self.ego_id, new_speed)

        # -------- LANE CHANGE CONTROL --------
        lane_cmd = action[1]
        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)

        if lane_cmd > self.lane_threshold:
            # change left
            self.sim.vehicle.changeLane(
                self.ego_id, lane_index + 1, self.lc_duration
            )

        elif lane_cmd < -self.lane_threshold:
            # change right
            self.sim.vehicle.changeLane(
                self.ego_id, lane_index - 1, self.lc_duration
            )

    def print_action(self, action):
        a, lc = action
        print(f"Accel: {a:.2f} m/s^2 | LaneCmd: {lc:.2f}")