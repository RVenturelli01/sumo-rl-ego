from abc import ABC, abstractmethod
from gymnasium.spaces import MultiDiscrete


class BaseObservationBuilder(ABC):

    @abstractmethod
    def build_obs(self):
        pass

    @abstractmethod
    def print_obs(self, obs):
        pass

    def reset(self):
        pass
    
    def set_sumo_simulation(self, sim):
        self.sim = sim

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id

        

class DefaultObservationBuilder(BaseObservationBuilder):
    
    def __init__(self):

        self.observation_space = MultiDiscrete([2])

    def build_obs(self):
        return self.observation_space.sample()
    
    def print_obs(self, obs):
        print("Default observation: ", obs)
    