from abc import ABC, abstractmethod
from gymnasium.spaces import MultiDiscrete


class BaseObservation(ABC):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def print_obs(self, obs):
        pass

    def setSumoSimulation(self, sim):
        self.sim = sim

    def setEgoId(self, ego_id):
        self.ego_id = ego_id

        

class DefaultObservation(BaseObservation):
    
    def __init__(self):

        self.observation_space = MultiDiscrete([2])

    def build(self):
        return self.observation_space.sample()
    
    def print_obs(self, obs):
        print("Default observation: ", obs)
    