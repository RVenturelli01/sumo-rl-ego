import numpy as np
from plugins.policies.base import BasePolicy

class MyPolicy(BasePolicy):

    def __init__(self):
        super().__init__()

    def predict(self, obs):
        ego_speed = obs[0]
        front_dist = obs[3] * 100
        front_rel_speed = obs[4] * 30
        front_left_dist = obs[5] * 100
        front_left_rel_speed = obs[6] * 30
        front_right_dist = obs[7] * 100
        front_right_rel_speed = obs[8] * 30
        left_free = bool(obs[1])
        right_free = bool(obs[2])

        # time to collision (TTC) = distance / relative speed
        ttc_front = front_dist / (-front_rel_speed) if front_rel_speed < 0 else float('inf')
        ttc_left = front_left_dist / (-front_left_rel_speed) if front_left_rel_speed < 0 else float('inf')
        ttc_right = front_right_dist / (-front_right_rel_speed) if front_right_rel_speed < 0 else float('inf')
        
        # Simple heuristic: 
        if left_free and ttc_left > 2.0:
            return 3  # lane change left
        
        if ttc_front < 2.0 or front_dist < 10.0:
            # if there's a car dangerously close in front, try to change lane
            if left_free and ttc_left > 2.0:
                return 3  # lane change left
            elif right_free and ttc_right > 2.0:
                return 4  # lane change right
            else:
                return 2  # decelerate
        
        if ttc_front < 4.0 or front_dist < 15.0:
            return 2  # decelerate
        
        if ttc_front > 6.0 and ego_speed < 25.0 :
            return 1  # accelerate
        
        return 0  # same speed