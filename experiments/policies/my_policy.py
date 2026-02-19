import numpy as np
from experiments.policies.base_policy import BasePolicy

class HeuristicPolicy(BasePolicy):

    def __init__(self, accel_action=2, brake_action=0, dist_idx=0, threshold=30):
        super().__init__()
        self.accel = accel_action
        self.brake = brake_action
        self.dist_idx = dist_idx
        self.threshold = threshold

    def predict(self, obs):
        dist = obs[self.dist_idx]
        if dist > self.threshold:
            return self.accel
        return self.brake
