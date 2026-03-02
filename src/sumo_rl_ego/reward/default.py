from sumo_rl_ego.reward.base import BaseRewardFunction


class DefaultRewardFunction(BaseRewardFunction):

    def compute(self, action, info):
        return 0.0

    def reset(self):
        pass
