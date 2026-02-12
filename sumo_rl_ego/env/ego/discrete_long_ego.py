import numpy as np
from gymnasium.spaces import Discrete
from enum import IntEnum
from .base_ego import BaseEgoVehicle


class DiscreteActions(IntEnum):
    N = 0      # no-op i.e. take no action
    D = 1      # Light slow down
    CSD_1 = 2  # 0.5 m/s^2
    CSD_2 = 3  # 1 m/s^2
    CSD_3 = 4  # 1.5 m/s^2
    HSD_1 = 5  # 2 m/s^2
    HSD_2 = 6  # 2.5 m/s^2
    HSD_3 = 7  # 3 m/s^2
    SS = 8     # Same speed


class DiscreteLongEgo(BaseEgoVehicle):

    def __init__(self, veh_id, time_step=0.1, delta_decel=0.5):
        super().__init__(veh_id)

        self.time_step = time_step
        self.delta_decel = delta_decel

        self.action_space = Discrete(len(DiscreteActions))

    def apply_action(self, sim, action: int):

        lane = sim.get_lane_index(self.id)
        speed = sim.get_speed(self.id)

        # -------- Light slow down --------
        if action == DiscreteActions.D:
            slowed_speed = max(0, speed - self.delta_decel * self.time_step)
            sim.set_speed(self.id, slowed_speed)

        # -------- Comfortable slow downs --------
        elif DiscreteActions.CSD_1 <= action <= DiscreteActions.CSD_3:
            decel = (action - DiscreteActions.CSD_1 + 1) * self.delta_decel
            slowed_speed = max(0, speed - decel * self.time_step)
            sim.set_speed(self.id, slowed_speed)

        # -------- Harsh slow downs --------
        elif DiscreteActions.HSD_1 <= action <= DiscreteActions.HSD_3:
            decel = (action - DiscreteActions.HSD_1 + 1) * self.delta_decel
            slowed_speed = max(0, speed - decel * self.time_step)
            sim.set_speed(self.id, slowed_speed)

        # -------- Same speed --------
        elif action == DiscreteActions.SS:
            sim.set_speed(self.id, speed)

        # -------- No-op (N) --------
        else:
            pass
