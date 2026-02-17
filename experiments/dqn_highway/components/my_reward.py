from sumo_rl_ego.reward.base_reward import BaseReward
import numpy as np


class MyReward(BaseReward):

    def __init__(
        self,
        target_speed: float = 30.0,
        w_speed: float = 5.0,
        w_offroad: float = -100.0,
        w_crash: float = -100.0,
        w_lane_change: float = 0.0,
        w_arrived: float = 1000.0
    ):
        self.target_speed = target_speed
        self.w_speed = w_speed
        self.w_offroad = w_offroad
        self.w_crash = w_crash
        self.w_lane_change = w_lane_change
        self.w_arrived = w_arrived


    def compute(self, action, info):

        if info["ep_status"]["truncated"] or info["ep_status"]["terminated"]:
            return self.compute_terminal(action, info)  

        reward = -1

        speed = self.sim.vehicle.getSpeed(self.ego_id)

        speed_error = abs(speed - self.target_speed)
        sigma = self.target_speed * 0.3
        speed_reward = np.exp(-(speed_error**2) / (2 * sigma**2))

        reward += self.w_speed * speed_reward
        reward = speed/10

        if action == 6 or action == 7:  # lane change
            reward += self.w_lane_change

        return reward


    def compute_terminal(self, action, info):

        reward = 0.0

        if info.get("collided", False):
            reward += self.w_crash

        elif info.get("off_road", False):
            reward += self.w_offroad

        elif info.get("arrived", False):
            max_steps = self.sim.config.max_steps
            step_count = info["ep_status"]["step_count"]
            reward += self.w_arrived * (max_steps - step_count) / max_steps

        return reward
    