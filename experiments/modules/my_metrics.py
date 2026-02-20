from sumo_rl_ego.metrics.base import BaseMetricsTracker
from statistics import mean, pstdev 



class MyMetrics(BaseMetricsTracker):
    def __init__(self, window=0):
        self.reset_global()
        self.reset()
        self.window = window

    # ======================
    # RESET
    # ======================
    def reset(self):
        # metriche episodio
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.ep_collisions = 0
        self.ep_offroad = 0
        self.ep_arrived = 0
        self.ep_lane_changes = 0
        self.speed_sum = 0.0

    def reset_global(self):
        # global stats
        self.total_episodes = 0
        self.global_rewards = []
        self.global_lengths = []
        self.global_avg_speeds = []
        self.global_success = []
        self.global_collisions = []
        self.global_offroad = []

    # ======================
    # STEP UPDATE
    # ======================
    def update_step(self, obs, action, reward, info):
        ego_status = info["ego_status"]

        self.ep_reward += reward
        self.ep_steps += 1

        if ego_status.get("collided", False):
            self.ep_collisions += 1

        if ego_status.get("off_road", False):
            self.ep_offroad += 1

        if ego_status.get("arrived", False):
            self.ep_arrived = 1

        if action in (3, 4):
            self.ep_lane_changes += 1

        # velocità
        if self.sim.ego_exists(self.ego_id):
            v = self.sim.vehicle.getSpeed(self.ego_id)
        else:
            v = 0.0

        self.speed_sum += v

        # snapshot step
        self._last_step_metrics = {
            "reward": reward,
            "speed": v,
        }


    # ======================
    # END EPISODE
    # ======================
    def end_episode(self, info):
        ep_status = info["ep_status"]

        avg_speed = self.speed_sum / self.ep_steps if self.ep_steps > 0 else 0.0

        # global counters
        self.total_episodes += 1
        self.global_rewards.append(self.ep_reward)
        self.global_lengths.append(self.ep_steps)
        self.global_avg_speeds.append(avg_speed)
        self.global_success.append(self.ep_arrived)
        self.global_collisions.append(self.ep_collisions)
        self.global_offroad.append(self.ep_offroad)

        # snapshot episodio
        self._last_episode_metrics = {
            "episode_reward": self.ep_reward,
            "episode_length": self.ep_steps,
            "episode_avg_speed": avg_speed,
            "episode_collisions": self.ep_collisions,
            "episode_offroad": self.ep_offroad,
            "episode_lane_changes": self.ep_lane_changes,
            "episode_arrived": self.ep_arrived,
            "episode_truncated": ep_status.get("truncated", False),
        }

        self.reset()

    # ======================
    # GETTERS
    # ======================
    def get_step_metrics(self):
        return getattr(self, "_last_step_metrics", {})

    def get_episode_metrics(self):
        return getattr(self, "_last_episode_metrics", {})

    def get_global_metrics(self):
        if self.total_episodes == 0:
            return {}

        # helper safe
        def safe_std(x):
            return pstdev(x) if len(x) > 1 else 0.0
        
        def slice(self, x):
            return x[-self.window:] if self.window > 0 else x

        return {
            "episodes": self.total_episodes,

            "ep_rew_mean": mean(slice(self, self.global_rewards)),
            "ep_rew_std": safe_std(slice(self, self.global_rewards)),

            "ep_length_mean": mean(slice(self, self.global_lengths)),
            "ep_length_std": safe_std(slice(self, self.global_lengths)),

            "ep_avg_speed_mean": mean(slice(self, self.global_avg_speeds)),
            "ep_avg_speed_std": safe_std(slice(self, self.global_avg_speeds)),

            "ep_success_rate": mean(slice(self, self.global_success)),

            "ep_collision_rate": mean(slice(self, self.global_collisions)),
            
            "ep_offroad_rate": mean(slice(self, self.global_offroad)),
        }


    # ======================
    # HISTORY EXPORT
    # ======================
    def get_global_history(self):
        """Per plot globali training"""
        return {
            "reward_per_episode": self.global_rewards,
            "length_per_episode": self.global_lengths,
            "avg_speed_per_episode": self.global_avg_speeds,
            "success_per_episode": self.global_success,
            "collisions_per_episode": self.global_collisions,
            "offroad_per_episode": self.global_offroad,
        }