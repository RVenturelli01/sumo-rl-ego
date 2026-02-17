from abc import ABC, abstractmethod


class BaseReward(ABC):

    @abstractmethod
    def compute(self, action, info):
        """
        Compute step reward when episode is ongoing.
        """
        pass

    def setSumoSimulation(self, sim):
        self.sim = sim

    def setEgoId(self, ego_id):
        self.ego_id = ego_id


class DefaultReward(BaseReward):
    
    def compute(self, action, info):
        return 0.0
    