from sumo_rl_ego.env.reward.base_reward import BaseReward
import numpy as np


class MyReward(BaseReward):

    def __init__(
        self,
        target_speed: float = 30.0,
        w_speed: float = 5.0,
        w_offroad: float = -1000.0,
        w_crash: float = -1000.0,
    ):
        self.target_speed = target_speed
        self.w_speed = w_speed
        self.w_offroad = w_offroad
        self.w_crash = w_crash


    def compute(self, action, info):

        if info["ep_status"]["truncated"] or info["ep_status"]["terminated"]:
            return self.compute_terminal(action, info)  

        reward = 0.0

        speed = self.sim.vehicle.getSpeed(self.ego_id)

        speed_error = abs(speed - self.target_speed)
        sigma = self.target_speed * 0.3
        speed_reward = np.exp(-(speed_error**2) / (2 * sigma**2))

        reward += self.w_speed * speed_reward
        reward = speed/10

        return reward


    def compute_terminal(self, action, info):

        reward = 0.0

        if info.get("collided", False):
            reward += self.w_crash

        elif info.get("off_road", False):
            reward += self.w_offroad


        return reward