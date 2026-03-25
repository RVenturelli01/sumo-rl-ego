from sumo_gym_ego import BaseRewardFunction
from sumo_gym_ego import EgoStatus


class TerminalReward(BaseRewardFunction):

    def __init__(
        self,
        w_crash=-1.0,
        w_offroad=-1.0,
        w_arrived=1.0,
        w_timeout=1.0,
    ):
        self.w_crash = w_crash
        self.w_offroad = w_offroad
        self.w_arrived = w_arrived
        self.w_timeout = w_timeout


    def compute(self, obs, action, next_obs, info):
        return 0.0

    def compute_terminal(self, obs, action, next_obs, info):

        if info["ego_status"] == EgoStatus.COLLIDED.value:
            return self.w_crash
        
        if info["ego_status"] == EgoStatus.TIMEOUT.value:
            return self.w_timeout

        if info["ego_status"] == EgoStatus.OFF_ROAD.value:
            return self.w_offroad

        if info["ego_status"] == EgoStatus.ARRIVED.value:
            return self.w_arrived

        return 0.0
