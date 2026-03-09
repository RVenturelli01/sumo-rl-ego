from statistics import mean, pstdev
from types import SimpleNamespace
from sumo_rl_ego.sumo_gym_ego.metrics.base import BaseMetricsTracker
from sumo_rl_ego.sumo_gym_ego.core.ego_status import EgoStatus
from sumo_rl_ego.plugins.egos.highway_discrete_ego import DiscreteActions


"""
Tracks episode-level metrics for the ego vehicle in SUMO RL.


Tracked metrics
---------------
- episode reward
- episode length
- average ego speed
- action usage rates (SS, ACC, DEC, LCL, LCR)
- termination causes (success, collision, off-road, timeout, etc.)


Metrics are accumulated during the episode and aggregated over
a rolling window of episodes for logging.


Windowing
---------
Statistics are computed over the last `window` episodes.
If `window <= 0`, the full history is used.
"""


class TerminationActionMetrics(BaseMetricsTracker):

    def __init__(self, window=100):
        self.window = window

        # hierarchical structure
        self.step = SimpleNamespace()
        self.episode = SimpleNamespace()
        self.history = SimpleNamespace()

        self.reset_global()
        self.reset()

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):
        """Reset episode accumulators."""

        self.episode.reward = 0.0
        self.episode.speed_sum = 0.0
        
        self.episode.action_ss_sum = 0.0
        self.episode.action_acc_sum = 0.0
        self.episode.action_dec_sum = 0.0
        self.episode.action_lcl_sum = 0.0
        self.episode.action_lcr_sum = 0.0

        self.step.metrics = {}

    def reset_global(self):
        """Reset global history."""

        self.history.total_episodes = 0

        self.history.rewards = []
        self.history.lengths = []
        self.history.avg_speeds = []

        self.history.success = []
        self.history.collision = []
        self.history.off_road = []
        self.history.truncated = []
        self.history.teleported = []
        self.history.removed_unknown = []

        self.history.action_ss_rate = []
        self.history.action_acc_rate = []
        self.history.action_dec_rate = []
        self.history.action_lcl_rate = []
        self.history.action_lcr_rate = []

    # =========================================================
    # STEP METRICS
    # =========================================================

    def compute_step_metrics(self, obs, action, next_obs, reward, info):

        ep = self.episode

        ep.reward += reward

        speed = self._get_current_speed()
        ep.speed_sum += speed

        ep.action_ss_sum += action == DiscreteActions.SS
        ep.action_acc_sum += action == DiscreteActions.ACC
        ep.action_dec_sum += action == DiscreteActions.DEC
        ep.action_lcl_sum += action == DiscreteActions.LCL
        ep.action_lcr_sum += action == DiscreteActions.LCR

        self.step.metrics = {
            "speed": speed,
        }

        return self.step.metrics

    # =========================================================
    # EPISODE FINALIZATION
    # =========================================================

    def finalize_episode_metrics(self, info):

        ep = self.episode
        hist = self.history
        
        event = info.get("event", "unknown")
        step_count = info.get("status", {}).get("step", 0)

        avg_speed = self._safe_div(ep.speed_sum, step_count)

        hist.total_episodes += 1

        hist.rewards.append(ep.reward)
        hist.lengths.append(step_count)
        hist.avg_speeds.append(avg_speed)

        hist.success.append(event == EgoStatus.ARRIVED.value)
        hist.collision.append(event == EgoStatus.COLLIDED.value)
        hist.off_road.append(event == EgoStatus.OFF_ROAD.value)
        hist.teleported.append(event == EgoStatus.TELEPORTED.value)
        hist.removed_unknown.append(event == EgoStatus.REMOVED_UNKNOWN.value)
        hist.truncated.append(event == "timeout")

        hist.action_ss_rate.append(self._safe_div(ep.action_ss_sum, step_count)) 
        hist.action_acc_rate.append(self._safe_div(ep.action_acc_sum, step_count)) 
        hist.action_dec_rate.append(self._safe_div(ep.action_dec_sum, step_count)) 
        hist.action_lcl_rate.append(self._safe_div(ep.action_lcl_sum, step_count)) 
        hist.action_lcr_rate.append(self._safe_div(ep.action_lcr_sum, step_count)) 
            
        metrics = {
            "ep_reward": ep.reward,
            "ep_length": step_count,
            "ep_avg_speed": avg_speed,
            "ep_ss_rate": hist.action_ss_rate[-1],
            "ep_acc_rate": hist.action_acc_rate[-1],
            "ep_dec_rate": hist.action_dec_rate[-1],
            "ep_lcl_rate": hist.action_lcl_rate[-1],
            "ep_lcr_rate": hist.action_lcr_rate[-1],
        }

        return metrics

    # =========================================================
    # LOGGING
    # =========================================================

    def get_log_metrics(self, window_override = None):

        hist = self.history

        if hist.total_episodes == 0:
            return {}

        w = window_override or self.window

        def win(x):
            return x[-w:] if w > 0 else x

        def safe_std(x):
            return pstdev(x) if len(x) > 1 else 0.0

        return {
            "feat/episodes": hist.total_episodes,

            "feat/ep_reward_mean": mean(win(hist.rewards)),
            "feat/ep_reward_std": safe_std(win(hist.rewards)),

            "feat/ep_length_mean": mean(win(hist.lengths)),
            "feat/ep_length_std": safe_std(win(hist.lengths)),

            "feat/ep_avg_speed_mean": mean(win(hist.avg_speeds)),
            "feat/ep_avg_speed_std": safe_std(win(hist.avg_speeds)),

            "events/success_rate": mean(win(hist.success)),
            "events/collision_rate": mean(win(hist.collision)),
            "events/offroad_rate": mean(win(hist.off_road)),
            "events/truncated_rate": mean(win(hist.truncated)),
            # "events/teleported_rate": mean(win(hist.teleported)),
            # "events/removed_unknown_rate": mean(win(hist.removed_unknown)),

            "action/ep_ss_rate": mean(win(hist.action_ss_rate)),
            "action/ep_acc_rate": mean(win(hist.action_acc_rate)),
            "action/ep_dec_rate": mean(win(hist.action_dec_rate)),
            "action/ep_lcl_rate": mean(win(hist.action_lcl_rate)),
            "action/ep_lcr_rate": mean(win(hist.action_lcr_rate)),
        }

    # =========================================================
    # HELPERS
    # =========================================================

    def _get_current_speed(self):
        if self.sim.ego_exists(self.ego_id):
            return self.sim.vehicle.getSpeed(self.ego_id)
        return 0.0

    @staticmethod
    def _safe_div(a, b):
        return a / b if b > 0 else 0.0
    
    from collections import defaultdict

    def print_log_metrics(self, window_override=None):
        metrics = self.get_log_metrics(window_override)
        window = window_override or self.window

        print(f"\n=== Metrics (last {window} episodes) ===")

        current_section = None

        for k, v in metrics.items():

            if "/" in k:
                section, name = k.split("/", 1)
            else:
                section, name = "other", k

            if section != current_section:
                print(f"\n{section}:")
                current_section = section

            print(f"    {name:<20} {v:8.2f}")