from enum import IntEnum
from gymnasium.spaces import Discrete
from sumo_rl_ego.sumo_gym_ego.ego.base import BaseEgoController


class DiscreteActions(IntEnum):
    N = 0  # no-op


class DefaultEgoController(BaseEgoController):

    def __init__(self):
        self.action_space = Discrete(len(DiscreteActions))

    def apply_action(self, action):
        pass

    def reset(self):
        pass

    def print_action(self, action):
        print(f"Action: {DiscreteActions(action).name}")
