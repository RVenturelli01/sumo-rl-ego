from abc import abstractmethod
from sumo_rl_ego.core.plugin import BaseEnvPlugin


class BaseObservationBuilder(BaseEnvPlugin):

    @abstractmethod
    def build_obs(self):
        pass