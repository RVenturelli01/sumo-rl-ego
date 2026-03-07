from sumo_rl_ego.sumo_gym_ego.reward.base import BaseRewardFunction


class HighwayBasicReward(BaseRewardFunction):

    def __init__(
        self,
        w_crash=-1.0,
        w_offroad=-1.0,
        w_arrived=1.0,
    ):
        self.w_crash = w_crash
        self.w_offroad = w_offroad
        self.w_arrived = w_arrived

        self.prev_pos = None
        self.prev_acc = 0.0

    def compute(self, obs, action, next_obs, info):
        if info["status"]["terminated"] or info["status"]["truncated"]:
            return self.compute_terminal(info["event"])
        return 0.0

    def compute_terminal(self, event):

        if event == "collided":
            return self.w_crash

        if event == "off_road":
            return self.w_offroad

        if event == "arrived":
            return self.w_arrived

        return 0.0

    def reset(self):
        self.prev_pos = None
        self.prev_acc = 0.0