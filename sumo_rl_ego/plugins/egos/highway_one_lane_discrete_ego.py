import random
from gymnasium.spaces import Discrete
from enum import IntEnum
from sumo_rl_ego.sumo_gym_ego.ego.base import BaseEgoController


class DiscreteActions(IntEnum):
    SS = 0     # same speed
    ACC = 1    # +1 m/s^2
    DEC = 2    # -2 m/s^2


class HighwayOneLaneDiscreteEgo(BaseEgoController):

    def __init__(self, 
                 acc_value=1.0,
                 dec_value=-2.0,
                 max_speed=50.0,):
    
        self.acc_value = acc_value
        self.dec_value = dec_value
        self.max_speed = max_speed

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
            new_speed = min(new_speed, self.max_speed)
            self.sim.vehicle.setSpeed(self.ego_id, new_speed)
            return

    def print_action(self, action):
        print(f"Action: {DiscreteActions(action).name}")