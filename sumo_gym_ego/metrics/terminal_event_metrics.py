from statistics import mean
from types import SimpleNamespace
from sumo_gym_ego import BaseMetricsTracker
from sumo_gym_ego import EgoStatus


class TerminalEventMetrics(BaseMetricsTracker):

    def __init__(self, window=100):
        self.window = window
        self.history = SimpleNamespace()
        self.reset_global()

    def reset(self):
        pass

    def reset_global(self):

        self.history.total_episodes = 0

        self.history.success = []
        self.history.collision = []
        self.history.off_road = []
        self.history.truncated = []

    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        return {}

    def finalize_episode_metrics(self, info):

        hist = self.history

        event = info.get("event", "unknown")

        hist.total_episodes += 1

        hist.success.append(event == EgoStatus.ARRIVED.value)
        hist.collision.append(event == EgoStatus.COLLIDED.value)
        hist.off_road.append(event == EgoStatus.OFF_ROAD.value)
        hist.truncated.append(event == "timeout")

        return {
            "event_success": hist.success[-1],
            "event_collision": hist.collision[-1],
        }

    def get_log_metrics(self):

        hist = self.history

        def win(x):
            return x[-self.window:] if self.window > 0 else x

        return {
            "events/success_rate": mean(win(hist.success)),
            "events/collision_rate": mean(win(hist.collision)),
            "events/offroad_rate": mean(win(hist.off_road)),
            "events/truncated_rate": mean(win(hist.truncated)),
        }