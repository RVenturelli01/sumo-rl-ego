from abc import ABC, abstractmethod
from enum import IntEnum
from gymnasium.spaces import Discrete


class BaseEgoController(ABC):

    @abstractmethod
    def apply_action(self, action):
        pass

    def reset(self):
        pass

    def set_sumo_simulation(self, sim, obs_builder=None, metrics=None, reward=None):
        self.sim = sim
        self.obs_builder = obs_builder
        self.metrics = metrics
        self.reward = reward

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id




class DiscreteActions(IntEnum):
    N = 0      # no-op


class DefaultEgoController(BaseEgoController):

    def __init__(self):
        self.action_space = Discrete(len(DiscreteActions))
    
    def print_action(self, action):
        print(f"Action: {DiscreteActions(action).name}")