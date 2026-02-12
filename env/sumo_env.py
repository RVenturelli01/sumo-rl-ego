import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from .simulation import SumoSimulation
from .ego_vehicle import EgoVehicle
from .observation import ObservationBuilder
from .reward import RewardFunction
from .done import DoneCondition
from configs.config import SumoConfig


class SumoEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: SumoConfig):
        super().__init__()

        self.config = config

        self.sim = SumoSimulation(config.build_cmd())
        self.ego = EgoVehicle(config.ego_id)
        self.obs_builder = ObservationBuilder()
        self.reward_fn = RewardFunction(config.target_speed)
        self.done_fn = DoneCondition(config.max_steps)

        self.action_space = self.ego.action_space
        self.observation_space = self.ego.observation_space


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.start()
        self.done_fn.reset()

        # step iniziale per spawn veicoli
        self.sim.step()

        obs = self.obs_builder.build(self.sim, self.ego)

        return obs, {}

    def step(self, action):
        accel = float(action[0])

        self.sim.apply_acceleration(self.ego.id, accel)
        self.sim.step()

        obs = self.obs_builder.build(self.sim, self.ego)
        reward = self.reward_fn.compute(self.sim, self.ego)
        done = self.done_fn.check(self.sim, self.ego)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        pass  # usa sumo-gui se vuoi visualizzare

    def close(self):
        self.sim.close()
