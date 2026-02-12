import numpy as np
from gymnasium.spaces import Box
from .base_observation import BaseObservation

class SimpleObservation(BaseObservation):

    def __init__(self):
        self.observation_space = Box(
            low=np.array([0.0, 0.0]),
            high=np.array([50.0, 100.0]),
            dtype=np.float32
        )

    def build(self, sim, ego):
        speed = sim.get_speed(ego.id)

        leader = sim.get_leader(ego.id)
        distance = 100.0
        if leader is not None:
            _, dist = leader
            distance = dist

        return np.array([speed, distance], dtype=np.float32)
