from gymnasium.spaces import Discrete
from enum import IntEnum
from .base_ego import BaseEgoVehicle


class DiscreteActions(IntEnum):
    N = 0      # no-op
    SS = 1     # same speed (explicit hold)
    ACC_1 = 2  # +1 m/s^2
    ACC_2 = 3  # +2 m/s^2
    DEC_1 = 4  # -1 m/s^2
    DEC_2 = 5  # -2 m/s^2


class DiscreteLongEgo(BaseEgoVehicle):

    def __init__(self, veh_id, time_step=0.1):
        super().__init__(veh_id)

        self.time_step = time_step
        self.action_space = Discrete(len(DiscreteActions))

    # ===================================
    # APPLY ACTION (TraCI-native)
    # ===================================
    def apply_action(self, sim, action: int):

        veh_id = self.id

        speed = sim.getSpeed(veh_id)

        # -------- No-op --------
        if action == DiscreteActions.N:
            return

        # -------- Same speed --------
        if action == DiscreteActions.SS:
            sim.setSpeed(veh_id, speed)
            return

        # -------- Accelerations --------
        if action == DiscreteActions.ACC_1:
            accel = 1.0
        elif action == DiscreteActions.ACC_2:
            accel = 2.0

        # -------- Decelerations --------
        elif action == DiscreteActions.DEC_1:
            accel = -1.0
        elif action == DiscreteActions.DEC_2:
            accel = -2.0
        else:
            return  # sicurezza

        new_speed = max(0.0, speed + accel * self.time_step)
        sim.setSpeed(veh_id, new_speed)
