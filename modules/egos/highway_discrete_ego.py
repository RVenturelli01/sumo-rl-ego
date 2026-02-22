import random
from gymnasium.spaces import Discrete
from enum import IntEnum
from sumo_rl_ego.ego.base import BaseEgoController


class DiscreteActions(IntEnum):
    SS = 0     # same speed
    ACC = 1    # +1 m/s^2 (default)
    DEC = 2    # -2 m/s^2 (default)
    LCL = 3    # lane change left
    LCR = 4    # lane change right


class HighwayDiscreteEgo(BaseEgoController):

    def __init__(self, 
                 acc_value=1.0,
                 dec_value=-2.0,
                 lc_duration=0):
    
        self.acc_value = acc_value
        self.dec_value = dec_value
        self.lc_duration = lc_duration

        self.action_space = Discrete(len(DiscreteActions))


    def apply_action(self, action):
        
        time_step = self.sim.config.time_step

        speed = self.sim.vehicle.getSpeed(self.ego_id)

        # SAME SPEED
        if action == DiscreteActions.SS:
            self.sim.vehicle.setSpeed(self.ego_id, speed)
            return

        # LONGITUDINAL CONTROL
        if action == DiscreteActions.ACC:
            accel = self.acc_value
        elif action == DiscreteActions.DEC:
            accel = self.dec_value
        else:
            accel = None

        if accel is not None:
            new_speed = max(0.0, speed + accel * time_step)
            self.sim.vehicle.setSpeed(self.ego_id, new_speed)
            return

        # LANE CHANGE
        lane_index = self.sim.vehicle.getLaneIndex(self.ego_id)

        # ---- change left ----
        if action == DiscreteActions.LCL:
            self.sim.vehicle.changeLane(self.ego_id, lane_index + 1, self.lc_duration)

        # ---- change right ----
        elif action == DiscreteActions.LCR:
            self.sim.vehicle.changeLane(self.ego_id, lane_index - 1, self.lc_duration)

    def print_action(self, action):
        print(f"Action: {DiscreteActions(action).name}")