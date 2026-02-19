from abc import ABC, abstractmethod
from enum import IntEnum
from gymnasium.spaces import Discrete


class BaseEgo(ABC):

    @abstractmethod
    def apply_action(self, action):
        pass

    @abstractmethod
    def print_action(self, action):
        pass

    def setSumoSimulation(self, sim):
        self.sim = sim

    def setEgoId(self, ego_id):
        self.ego_id = ego_id




class DiscreteActions(IntEnum):
    N = 0      # no-op


class DefaultEgo(BaseEgo):

    def __init__(self):
        self.action_space = Discrete(len(DiscreteActions))

    def apply_action(self, action):
        return
    
    def print_action(self, action):
        return DiscreteActions(action).name