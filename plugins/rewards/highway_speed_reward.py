from sumo_rl_ego.reward.base import BaseRewardFunction
import numpy as np


class HighwaySpeedReward(BaseRewardFunction):

    def __init__(
        self,
        max_speed=30.0,
        w_speed=1.0,
        w_crash=-1.0,
        w_offroad=-1.0,
        w_arrived=1.0,
        w_lane_change=-0.05,
    ):
        self.max_speed = max_speed
        self.w_speed = w_speed
        self.w_crash = w_crash
        self.w_offroad = w_offroad
        self.w_arrived = w_arrived
        self.w_lane_change = w_lane_change

    def compute(self, action, info):
        if info["ep_status"]["truncated"] or info["ep_status"]["terminated"]:
            return self.compute_terminal(info)

        reward = 0.0

        # progress reward (core signal)
        speed = self.sim.vehicle.getSpeed(self.ego_id)
        reward += self.w_speed * np.clip(speed / self.max_speed, 0.0, 1.0)

        # small lane change penalty (anti zig-zag)
        if action in (3, 4):
            reward += self.w_lane_change

        return reward

    def compute_terminal(self, info):
        if info["ego_status"]["collided"]:
            return self.w_crash

        if info["ego_status"]["off_road"]:
            return self.w_offroad

        if info["ego_status"]["arrived"]:
            return self.w_arrived

        return 0.0