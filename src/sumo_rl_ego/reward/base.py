from abc import ABC, abstractmethod


class BaseRewardFunction(ABC):

    @abstractmethod
    def compute(self, action, info):
        pass

    def reset(self):
        pass

    def set_sumo_simulation(self, sim):
        self.sim = sim

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id



class DefaultRewardFunction(BaseRewardFunction):
    
    def compute(self, action, info):
        return 0.0
    