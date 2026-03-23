from abc import ABC, abstractmethod


class BaseEnvPlugin(ABC):
    """Common base for all environment plugins (ego, observation, reward, metrics)."""

    def reset(self):
        pass

    def bind(self, config, sim, sim_bus):
        self.config = config
        self.sim = sim
        self.ego_id = config.ego_id
        self.sim_bus = sim_bus
    

    

class BaseEgoController(BaseEnvPlugin):

    @abstractmethod
    def apply_action(self, action):
        pass




class BaseObservationBuilder(BaseEnvPlugin):

    @abstractmethod
    def build_obs(self):
        pass




class BaseRewardFunction(BaseEnvPlugin):

    @abstractmethod
    def compute(self, obs, action, next_obs, info):
        pass

    @abstractmethod
    def compute_terminal(self, obs, action, next_obs, info):
        pass




class BaseMetricsTracker(BaseEnvPlugin):

    @abstractmethod
    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        pass

    @abstractmethod
    def compute_episode_metrics(self, obs, action, next_obs, reward, info):
        pass

