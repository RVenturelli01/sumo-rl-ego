from gymnasium.spaces import Discrete
from enum import IntEnum
from ....sumo_rl_ego.ego.base_ego import BaseEgoVehicle


class DiscreteActions(IntEnum):
    N = 0      # no-op
    SS = 1     # same speed
    ACC_1 = 2  # +1 m/s^2
    ACC_2 = 3  # +2 m/s^2
    DEC_1 = 4  # -1 m/s^2
    DEC_2 = 5  # -1 m/s^2
    LCL = 6    # lane change left
    LCR = 7    # lane change right


class MyEgo(BaseEgoVehicle):

    def __init__(self, 
                 lc_duration=0):
    
        self.lc_duration = lc_duration

        self.action_space = Discrete(len(DiscreteActions))


    def apply_action(self, action):
        
        time_step = self.sim.config.time_step

        speed = self.sim.vehicle.getSpeed(self.ego_id)

        # NO-OP
        if action == DiscreteActions.N:
            return

        # SAME SPEED
        if action == DiscreteActions.SS:
            self.sim.vehicle.setSpeed(self.id, speed)
            return

        # LONGITUDINAL CONTROL
        if action == DiscreteActions.ACC_1:
            accel = 1.0
        elif action == DiscreteActions.ACC_2:
            accel = 2.0
        elif action == DiscreteActions.DEC_1:
            accel = -1.0
        elif action == DiscreteActions.DEC_2:
            accel = -2.0
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
