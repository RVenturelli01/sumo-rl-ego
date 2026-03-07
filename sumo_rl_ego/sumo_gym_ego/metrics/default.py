from sumo_rl_ego.sumo_gym_ego.metrics.base import BaseMetricsTracker


class DefaultMetricsTracker(BaseMetricsTracker):

    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        pass

    def finalize_episode_metrics(self, info):
        pass

    def get_log_metrics(self):
        pass

    def reset(self):
        pass
