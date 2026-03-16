import numpy as np
from sumo_gym_ego import BaseRewardFunction



class StepPenalty(BaseRewardFunction):

    def __init__(self, penalty=-1.0):
        self.penalty = penalty  

    def compute(self, obs, action, next_obs, info):
        return self.penalty

    def compute_terminal(self, obs, action, next_obs, info):
        return 0.0