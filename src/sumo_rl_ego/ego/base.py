from abc import ABC, abstractmethod
from enum import IntEnum
from gymnasium.spaces import Discrete


class BaseEgoController(ABC):

    @abstractmethod
    def apply_action(self, action):
        pass

    @abstractmethod
    def print_action(self, action):
        pass

    def reset(self):
        pass

    def set_sumo_simulation(self, sim):
        self.sim = sim

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id




class DiscreteActions(IntEnum):
    N = 0      # no-op


class DefaultEgoController(BaseEgoController):

    def __init__(self):
        self.action_space = Discrete(len(DiscreteActions))

    def apply_action(self, action):
        return
    
    def print_action(self, action):
        print(f"Action: {DiscreteActions(action).name}")