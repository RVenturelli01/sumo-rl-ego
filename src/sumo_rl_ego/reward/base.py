from abc import abstractmethod
from sumo_rl_ego.core.plugin import BaseEnvPlugin


class BaseRewardFunction(BaseEnvPlugin):

    @abstractmethod
    def compute(self, action, info):
        pass