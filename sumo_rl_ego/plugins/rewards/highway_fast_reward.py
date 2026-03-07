import numpy as np
from sumo_rl_ego.plugins.rewards.highway_basic_reward import HighwayBasicReward
from sumo_rl_ego.plugins.rewards.components import*


class HighwayFastReward(HighwayBasicReward):

    def __init__(
        self,
        max_speed=50.0,
        w_progress=0.5,
        w_step=-0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_speed = max_speed
        self.w_progress = w_progress
        self.w_step = w_step

    def compute(self, obs, action, next_obs, info):

        if info["status"]["terminated"] or info["status"]["truncated"]:
            return self.compute_terminal(info["event"])

        reward = 0.0

        reward += self.w_step

        pos = self.sim.vehicle.getPosition(self.ego_id)[0]
        time_step = self.config.time_step

        if self.prev_pos is not None:
            delta_x = pos - self.prev_pos
            reward += progress_reward(delta_x, self.max_speed*time_step, self.w_progress)

        self.prev_pos = pos

        return reward