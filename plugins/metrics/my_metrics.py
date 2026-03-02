from statistics import mean, pstdev
from sumo_rl_ego.metrics.base import BaseMetricsTracker


class MyMetrics(BaseMetricsTracker):
    def __init__(self, window=0):
        self.window = window
        self.reset_global()
        self.reset()

    # ======================
    # RESET
    # ======================
    def reset(self):
        # Episodic state
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.speed_sum = 0.0

        #: termination flags
        self.ep_arrived = 0
        self.ep_collided = 0
        self.ep_offroad = 0
        self.ep_truncated = 0
        self.ep_teleported = 0
        self.ep_removed_unknown = 0

        self._last_step_metrics = {}
        self._last_episode_metrics = {}

    def reset_global(self):
        self.total_episodes = 0

        self.global_rewards = []
        self.global_lengths = []
        self.global_avg_speeds = []

        self.global_success = []
        self.global_collisions = []
        self.global_offroad = []
        self.global_truncated = []
        self.global_teleported = []
        self.global_removed_unknown = []

    # ======================
    # STEP METRICS
    # ======================
    def compute_step_metrics(self, obs, action, reward, info):
        ego_status = info.get("ego_status", {})
        ep_status = info.get("ep_status", {})

        self.ep_steps += 1
        self.ep_reward += reward

        # --- termination causes
        self.ep_arrived |= int(ego_status.get("arrived", False))
        self.ep_collided += int(ego_status.get("collided", False))
        self.ep_offroad += int(ego_status.get("off_road", False))
        self.ep_teleported += int(ego_status.get("teleported", False))
        self.ep_removed_unknown += int(ego_status.get("removed_unknown", False))
        self.ep_truncated += int(ep_status.get("truncated", False))

        # --- speed tracking
        v = self._get_current_speed()
        self.speed_sum += v

        self._last_step_metrics = {
            "reward": reward,
            "speed": v,
        }
        return self._last_step_metrics

    # ======================
    # EPISODE FINALIZATION
    # ======================
    def finalize_episode_metrics(self):
        avg_speed = self._safe_div(self.speed_sum, self.ep_steps)

        # --- update global stats
        self.total_episodes += 1
        self.global_rewards.append(self.ep_reward)
        self.global_lengths.append(self.ep_steps)
        self.global_avg_speeds.append(avg_speed)

        self.global_success.append(self.ep_arrived)
        self.global_collisions.append(self.ep_collided)
        self.global_offroad.append(self.ep_offroad)
        self.global_truncated.append(self.ep_truncated)
        self.global_teleported.append(self.ep_teleported)
        self.global_removed_unknown.append(self.ep_removed_unknown)

        self._last_episode_metrics = {
            "ep_reward": self.ep_reward,
            "ep_length": self.ep_steps,
            "ep_avg_speed": avg_speed,
            "ep_success": self.ep_arrived,
            "ep_collisions": self.ep_collided,
            "ep_offroad": self.ep_offroad,
            "ep_truncated": self.ep_truncated,
            "ep_teleported": self.ep_teleported,
            "ep_removed_unknown": self.ep_removed_unknown,
        }

        self.reset()
        return self._last_episode_metrics

    # ======================
    # LOGGING METRICS
    # ======================
    def get_log_metrics(self):
        if self.total_episodes == 0:
            return {}

        w = self.window

        def win(x):
            return x[-w:] if w > 0 else x

        def safe_std(x):
            return pstdev(x) if len(x) > 1 else 0.0

        return {
            # volume
            "feat/episodes": self.total_episodes,

            # reward
            "feat/ep_reward_mean": mean(win(self.global_rewards)),
            "feat/ep_reward_std": safe_std(win(self.global_rewards)),

            # length
            "feat/ep_length_mean": mean(win(self.global_lengths)),
            "feat/ep_length_std": safe_std(win(self.global_lengths)),

            # dynamics
            "feat/ep_avg_speed_mean": mean(win(self.global_avg_speeds)),

            # termination causes
            "cause/success_rate": mean(win(self.global_success)),
            "cause/collision_rate": mean(win(self.global_collisions)),
            "cause/offroad_rate": mean(win(self.global_offroad)),
            "cause/truncated_rate": mean(win(self.global_truncated)),
            "cause/teleported_rate": mean(win(self.global_teleported)),
            "cause/removed_unknown_rate": mean(win(self.global_removed_unknown)),
        }

    # ======================
    # HELPERS
    # ======================
    def _get_current_speed(self):
        if hasattr(self, "sim") and self.sim.ego_exists(self.ego_id):
            return self.sim.vehicle.getSpeed(self.ego_id)
        return 0.0

    @staticmethod
    def _safe_div(a, b):
        return a / b if b > 0 else 0.0
