from sumo_rl_ego.plugins.rewards.highway_basic_reward import HighwayBasicReward
from sumo_rl_ego.plugins.rewards.components import*


class HighwaySmoothReward(BaseHighwayReward):

    def __init__(
        self,
        target_speed=20.0,
        max_speed=50.0,
        w_progress=0.3,
        w_speed=-0.4,
        w_jerk=-0.2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.target_speed = target_speed
        self.max_speed = max_speed
        self.w_progress = w_progress
        self.w_speed = w_speed
        self.w_jerk = w_jerk

    def compute(self, obs, action, next_obs, info):

        if info["status"]["terminated"] or info["status"]["truncated"]:
            return self.compute_terminal(info["event"])

        reward = 0.0

        speed = self.sim.vehicle.getSpeed(self.ego_id)
        acc = self.sim.vehicle.getAcceleration(self.ego_id)
        pos = self.sim.vehicle.getPosition(self.ego_id)[0]

        if self.prev_pos is not None:
            delta_x = pos - self.prev_pos
            reward += progress_reward(delta_x, self.max_speed, self.w_progress)

        self.prev_pos = pos

        reward += speed_tracking(speed, self.target_speed, self.w_speed)

        reward += jerk_penalty(acc, self.prev_acc, self.w_jerk)

        self.prev_acc = acc

        return reward