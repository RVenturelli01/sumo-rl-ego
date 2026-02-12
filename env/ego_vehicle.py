from gymnasium import spaces
import numpy as np


class EgoVehicle:
    def __init__(self, veh_id="ego"):
        self.id = veh_id

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([50.0, 200.0], dtype=np.float32),
            dtype=np.float32
        )
