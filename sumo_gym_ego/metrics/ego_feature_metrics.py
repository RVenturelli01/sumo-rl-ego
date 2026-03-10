from statistics import mean, pstdev
from types import SimpleNamespace
from sumo_gym_ego import BaseMetricsTracker


class EgoFeatureMetrics(BaseMetricsTracker):

    def __init__(self, window=100):
        self.window = window

        self.episode = SimpleNamespace()
        self.history = SimpleNamespace()

        self.reset_global()
        self.reset()

    def reset(self):

        self.episode.reward = 0.0
        self.episode.speed_sum = 0.0

    def reset_global(self):

        self.history.rewards = []
        self.history.lengths = []
        self.history.avg_speeds = []

    def compute_step_metrics(self, obs, action, next_obs, reward, info):

        ep = self.episode

        ep.reward += reward

        speed = self._get_current_speed()
        ep.speed_sum += speed

        return {"speed": speed}

    def finalize_episode_metrics(self, info):

        ep = self.episode
        hist = self.history

        step_count = info.get("status", {}).get("step", 0)

        avg_speed = self._safe_div(ep.speed_sum, step_count)

        hist.rewards.append(ep.reward)
        hist.lengths.append(step_count)
        hist.avg_speeds.append(avg_speed)

        return {
            "ep_reward": ep.reward,
            "ep_length": step_count,
            "ep_avg_speed": avg_speed,
        }

    def get_log_metrics(self):

        hist = self.history

        def win(x):
            return x[-self.window:] if self.window > 0 else x

        def safe_std(x):
            return pstdev(x) if len(x) > 1 else 0.0

        return {
            "feat/ep_reward_mean": mean(win(hist.rewards)),
            "feat/ep_reward_std": safe_std(win(hist.rewards)),
            "feat/ep_length_mean": mean(win(hist.lengths)),
            "feat/ep_avg_speed_mean": mean(win(hist.avg_speeds)),
        }

    def _get_current_speed(self):
        if self.sim.ego_exists(self.ego_id):
            return self.sim.vehicle.getSpeed(self.ego_id)
        return 0.0

    @staticmethod
    def _safe_div(a, b):
        return a / b if b > 0 else 0.0