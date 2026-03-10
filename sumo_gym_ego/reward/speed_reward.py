import numpy as np
from sumo_gym_ego import BaseRewardFunction



class SpeedReward(BaseRewardFunction):

    def __init__(self, max_speed=50.0, weight=1.0):
        self.max_speed = max_speed
        self.weight = weight

    def compute(self, obs, action, next_obs, info):

        v = self.sim.vehicle.getSpeed(self.ego_id)

        v = np.clip(v, 0, self.max_speed)

        return self.weight * (v / self.max_speed)

    def compute_terminal(self, obs, action, next_obs, info):
        return 0.0