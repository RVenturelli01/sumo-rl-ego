from sumo_rl_ego.reward.base_reward import BaseReward
import numpy as np


class MyReward(BaseReward):

    def __init__(
        self,
        w_speed=5.0,
        w_crash=-100.0,
        w_offroad=-100.0,
        w_lane_change=-0.2,
        w_step=-1,
        w_arrived=100.0,
    ):
        self.w_speed = w_speed
        self.w_crash = w_crash
        self.w_offroad = w_offroad
        self.w_lane_change = w_lane_change
        self.w_step = w_step
        self.w_arrived = w_arrived


    def compute(self, action, info):
        if info["ep_status"]["truncated"] or info["ep_status"]["terminated"]:
            return self.compute_terminal(info)

        reward = self.w_step

        # progress reward
        speed = self.sim.vehicle.getSpeed(self.ego_id)
        reward += self.w_speed*speed/30  # normalize by a reasonable max speed

        # lane change penalty
        if action in (3, 4):
            reward += self.w_lane_change

        return reward

    def compute_terminal(self, info):
        reward = 0.0

        if info["ego_status"]["collided"]:
            reward += self.w_crash

        elif info["ego_status"]["off_road"]:
            reward += self.w_offroad

        elif info["ego_status"]["arrived"]:
            reward += self.w_arrived

        return reward
