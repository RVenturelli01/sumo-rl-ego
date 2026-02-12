import numpy as np
from gymnasium.spaces import Box
from .base_ego import BaseEgoVehicle

class LongitudinalEgo(BaseEgoVehicle):

    def __init__(self, veh_id):
        super().__init__(veh_id)
        self.action_space = Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)

    def apply_action(self, sim, action):
        accel = float(action[0])
        sim.set_acceleration(self.id, accel)
