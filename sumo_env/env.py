import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci

class SumoEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.sumo_cmd = [
            "sumo-gui",
            "-c", "networks/highway_fast/highway.sumocfg"
        ]

        # Azione: accelerazione target
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Osservazione: velocità ego
        self.observation_space = spaces.Box(
            low=0.0,
            high=50.0,
            shape=(1,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        traci.start(self.sumo_cmd)
        traci.simulationStep()

        speed = traci.vehicle.getSpeed("ego")

        return np.array([speed], dtype=np.float32), {}

    def step(self, action):

        accel = action[0]

        # convertiamo accelerazione in speed target
        current_speed = traci.vehicle.getSpeed("ego")
        new_speed = max(0, current_speed + accel)

        traci.vehicle.setSpeed("ego", new_speed)

        traci.simulationStep()

        speed = traci.vehicle.getSpeed("ego")

        reward = speed  # reward banale: vai più veloce = meglio

        terminated = traci.simulation.getTime() >= 100
        truncated = False

        return np.array([speed], dtype=np.float32), reward, terminated, truncated, {}

    def close(self):
        traci.close()
