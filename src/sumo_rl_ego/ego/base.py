from abc import abstractmethod
from sumo_rl_ego.core.plugin import BaseEnvPlugin


class BaseEgoController(BaseEnvPlugin):

    @abstractmethod
    def apply_action(self, action):
        pass