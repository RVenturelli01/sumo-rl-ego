import numpy as np
from gymnasium.spaces import Box
from .base_ego import BaseEgoVehicle


class ContinuousLongEgo(BaseEgoVehicle):

    def __init__(self, veh_id, min_acc=-3.0, max_acc=3.0):
        super().__init__(veh_id)

        assert min_acc < max_acc, "min_acc must be < max_acc"

        self.min_acc = float(min_acc)
        self.max_acc = float(max_acc)

        self.action_space = Box(
            low=np.array([self.min_acc], dtype=np.float32),
            high=np.array([self.max_acc], dtype=np.float32),
            dtype=np.float32
        )

    def apply_action(self, sim, action):
        accel = float(action[0])

        # Clip acceleration
        accel = np.clip(accel, self.min_acc, self.max_acc)

        sim.set_acceleration(self.id, accel)
