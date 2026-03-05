from abc import abstractmethod
from sumo_rl_ego.sumo_gym_ego.core.plugin import BaseEnvPlugin


class BaseMetricsTracker(BaseEnvPlugin):

    @abstractmethod
    def compute_step_metrics(self, obs, action, reward, info):
        pass

    @abstractmethod
    def finalize_episode_metrics(self):
        pass

    @abstractmethod
    def get_log_metrics(self):
        pass