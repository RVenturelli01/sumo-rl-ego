import traci
from gymnasium.spaces import Discrete
from enum import IntEnum
from .base_ego import BaseEgoVehicle


class DiscreteActions(IntEnum):
    N = 0      # no-op
    SS = 1     # same speed
    ACC_1 = 2  # +1 m/s^2
    ACC_2 = 3  # +2 m/s^2
    DEC_1 = 4  # -1 m/s^2
    DEC_2 = 5  # -1 m/s^2
    LCL = 6    # lane change left
    LCR = 7    # lane change right


class DiscreteLongLatEgo(BaseEgoVehicle):

    def __init__(self, veh_id, time_step=0.1, lc_duration=2.0):
        super().__init__(veh_id)

        self.time_step = time_step
        self.lc_duration = lc_duration
        self.action_space = Discrete(len(DiscreteActions))

    # ===================================
    # APPLY ACTION
    # ===================================
    def apply_action(self, sim, action: int):
        
        veh_id = self.id

        speed = sim.getSpeed(veh_id)

        # NO-OP
        if action == DiscreteActions.N:
            return

        # SAME SPEED
        if action == DiscreteActions.SS:
            sim.setSpeed(veh_id, speed)
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
            new_speed = max(0.0, speed + accel * self.time_step)
            sim.setSpeed(veh_id, new_speed)
            return

        # LANE CHANGE
        lane_index = sim.getLaneIndex(veh_id)

        # ---- change left ----
        if action == DiscreteActions.LCL:
            sim.changeLane(veh_id, lane_index + 1, self.lc_duration)

        # ---- change right ----
        elif action == DiscreteActions.LCR:
            sim.changeLane(veh_id, lane_index - 1, self.lc_duration)
