import numpy as np
from src.infra.policy.base import BasePolicy

'''
Policy based on observation schema:
- ego speed (normalized)
- distance front same lane (normalized)
- ttc front same lane (normalized)
- ttc front left (normalized)
- ttc front right (normalized)
- lane index (normalized)
- left lane free (binary)
- right lane free (binary)

Policy based on ego logic:
    SS = 0     # same speed
    ACC = 1    # +1 m/s^2 (default)
    DEC = 2    # -2 m/s^2 (default)
    LCL = 3    # lane change left
    LCR = 4    # lane change right

'''

class MyPolicy(BasePolicy):

    def __init__(self,
                 max_speed=50.0,
                 max_ttc=100,
                 max_distance=100.0,
                 lane_gap=5.0,
                 acc_value=1.0,
                 dec_value=-4.0
                 ):
        super().__init__()

        self.max_speed = max_speed
        self.max_ttc = max_ttc
        self.max_distance = max_distance
        self.lane_gap = lane_gap
        self.acc_value = acc_value
        self.dec_value = dec_value

    def predict(self, obs):
        ego_speed = obs[0] * self.max_speed
        front_dist = obs[1] * self.max_distance
        ttc_front = obs[2] * self.max_ttc
        ttc_left = obs[3] * self.max_ttc
        ttc_right = obs[4] * self.max_ttc
        lane_index = obs[5]
        left_free = obs[6] 
        right_free = obs[7] 


        rel_front_speed = ego_speed - (front_dist / ttc_front) if (ttc_front<100 or front_dist<100) else 0
 
        # can i brake before the front vehicle?
        safe_dist = rel_front_speed**2 / (2 * abs(self.dec_value)) if rel_front_speed > 0 else 0

        # heuristic
        if front_dist < safe_dist:
            if ttc_left > ttc_right and left_free:
                return 3 # LCL
            elif right_free:
                return 4 # LCR
            return 3 # LCL (turn left and pray)
        
        if left_free and ttc_left > 4:
            return 3 # LCL
        
        if front_dist < safe_dist * 2:
            return 2 # DEC
        
        if ego_speed > self.max_speed*0.8:
            return 0 # SS

        return 1 # ACC
