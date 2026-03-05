from abc import abstractmethod
from sumo_rl_ego.sumo_gym_ego.core.plugin import BaseEnvPlugin


class BaseObservationBuilder(BaseEnvPlugin):

    @abstractmethod
    def build_obs(self):
        pass