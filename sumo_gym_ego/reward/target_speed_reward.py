from sumo_gym_ego import BaseRewardFunction


class TargetSpeedReward(BaseRewardFunction):

    def __init__(self, target_speed=20.0, weight=1.0):
        self.target_speed = target_speed
        self.weight = weight

    def compute(self, obs, action, next_obs, info):

        v = self.sim.vehicle.getSpeed(self.ego_id)

        diff_v = v - self.target_speed

        return - self.weight * (diff_v / self.target_speed)**2

    def compute_terminal(self, obs, action, next_obs, info):
        return 0.0