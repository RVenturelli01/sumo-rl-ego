from abc import ABC, abstractmethod

class BaseMetricsTracker(ABC):

    @abstractmethod
    def update_step(self, obs, action, reward, info):
        pass

    @abstractmethod
    def get_step_metrics(self):
        pass

    @abstractmethod
    def end_episode(self, info):
        pass

    @abstractmethod
    def get_episode_metrics(self):
        pass

    @abstractmethod
    def get_global_metrics(self):
        pass
        
    def reset(self):
        pass
        
    def set_sumo_simulation(self, sim):
        self.sim = sim

    def set_ego_id(self, ego_id):
        self.ego_id = ego_id



class DefaultMetricsTracker(BaseMetricsTracker):

    def update_step(self, obs, action, reward, info):
        pass

    def get_step_metrics(self):
        return {}
    
    def end_episode(self, info):
        pass

    def get_episode_metrics(self):
        return {}
    
    def get_global_metrics(self):
        return {}

    def get_metrics_step(self):
        return {}

    def end_episode(self, info):
        pass

    def get_metrics_episode(self):
        return {}
    
    def get_global_metrics(self):
        return {}
    