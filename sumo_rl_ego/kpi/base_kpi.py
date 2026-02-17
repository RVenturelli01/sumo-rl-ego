from abc import ABC, abstractmethod

class BaseKPI(ABC):

    @abstractmethod
    def compute_kpis(self):
        pass

    @abstractmethod
    def compute_episode_kpis(self):
        pass

    def setSumoSimulation(self, sim):
        self.sim = sim

    def setEgoId(self, ego_id):
        self.ego_id = ego_id



class DefaultKPI(BaseKPI):

    def compute_kpis(self):
        return {}

    def compute_episode_kpis(self):
        return {}