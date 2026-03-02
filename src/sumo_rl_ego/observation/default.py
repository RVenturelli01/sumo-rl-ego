from gymnasium.spaces import MultiDiscrete
from sumo_rl_ego.observation.base import BaseObservationBuilder


class DefaultObservationBuilder(BaseObservationBuilder):

    def __init__(self):
        self.observation_space = MultiDiscrete([2])

    def build_obs(self):
        return self.observation_space.sample()

    def reset(self):
        pass

    def print_obs(self, obs):
        print("Default observation:", obs)
