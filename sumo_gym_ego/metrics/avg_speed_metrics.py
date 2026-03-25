from statistics import mean, pstdev
from types import SimpleNamespace
from sumo_gym_ego import BaseMetricsTracker


class AvgSpeedMetrics(BaseMetricsTracker):

    def __init__(self):
        self.reset()

    def reset(self):
        self.reward_sum = 0.0
        self.speed_sum = 0.0
        self.step_count = 0


    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        speed = self._get_current_speed()
        
        self.speed_sum += speed
        self.step_count += 1

        return {}


    def compute_episode_metrics(self, obs, action, next_obs, reward, info):


        avg_speed = self.speed_sum / self.step_count if self.step_count > 0 else 0.0

        return {
            "performance/ep_avg_speed": avg_speed,
        }


    def _get_current_speed(self):
        if self.sim.ego_exists(self.ego_id):
            return self.sim.vehicle.getSpeed(self.ego_id)
        return 0.0
