from statistics import mean, pstdev
from types import SimpleNamespace
from sumo_gym_ego import BaseMetricsTracker


class PerformanceMetrics(BaseMetricsTracker):

    def __init__(self):
        self.reset()

    def reset(self):
        self.reward_sum = 0.0
        self.speed_sum = 0.0


    def compute_step_metrics(self, obs, action, next_obs, reward, info):

        self.reward_sum += reward

        speed = self._get_current_speed()
        self.speed_sum += speed

        return {}


    def compute_episode_metrics(self, obs, action, next_obs, reward, info):

        step_count = info.get("step", 0)

        avg_speed = self.speed_sum / step_count if step_count > 0 else 0.0

        return {
            "performance/return": self.reward_sum,
            "performance/speed_mean": avg_speed,
        }


    def _get_current_speed(self):
        if self.sim.ego_exists(self.ego_id):
            return self.sim.vehicle.getSpeed(self.ego_id)
        return 0.0
