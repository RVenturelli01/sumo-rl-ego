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
        self.ep_lane_changes = 0
        self.speed_sum = 0.0

        self.ep_actions_SS = 0
        self.ep_actions_ACC = 0
        self.ep_actions_DEC = 0
        self.ep_actions_LCL = 0
        self.ep_actions_LCR = 0

        self.ep_arrived = 0
        self.ep_collided = 0
        self.ep_offroad = 0
        self.teleported = 0
        self.removed_unknown = 0
        self.ep_truncated = 0

    def reset_global(self):
        # global stats
        self.total_episodes = 0

        self.global_rewards = []
        self.global_lengths = []
        self.global_avg_speeds = []
        self.global_lane_changes = []

        self.global_action_SS_rate = []
        self.global_action_ACC_rate = []
        self.global_action_DEC_rate = []
        self.global_action_LCL_rate = []
        self.global_action_LCR_rate = []    

        self.global_success = []
        self.global_collisions = []
        self.global_offroad = []
        self.global_truncated = []
        self.global_teleported = []
        self.global_removed_unknown = []    

    # ======================
    # STEP UPDATE
    # ======================
    def update_step(self, obs, action, reward, info):
        ego_status = info["ego_status"]

        self.ep_reward += reward
        self.ep_steps += 1

        if ego_status.get("arrived", False):
            self.ep_arrived = 1

        if ego_status.get("collided", False):
            self.ep_collided += 1

        if ego_status.get("off_road", False):
            self.ep_offroad += 1

        if ego_status.get("teleported", False):
            self.teleported += 1

        if ego_status.get("removed_unknown", False):
            self.removed_unknown += 1

        # if action in (3, 4):
        #     self.ep_lane_changes += 1

        # if action == 0:
        #     self.ep_actions_SS += 1
        # elif action == 1:
        #     self.ep_actions_ACC += 1
        # elif action == 2:
        #     self.ep_actions_DEC += 1
        # elif action == 3:
        #     self.ep_actions_LCL += 1
        # elif action == 4:
        #     self.ep_actions_LCR += 1

        # velocità
        if self.sim.ego_exists(self.ego_id):
            v = self.sim.vehicle.getSpeed(self.ego_id)
        else:
            v = 0.0

        self.speed_sum += v

        # snapshot step
        self._last_step_metrics = {
            "action": action,
            "reward": reward,
            "speed": v,
        }


    # ======================
    # END EPISODE
    # ======================
    def end_episode(self, info):
        ep_status = info["ep_status"]
        
        if ep_status.get("truncated", False):
            self.ep_truncated = 1

        avg_speed = self.speed_sum / self.ep_steps if self.ep_steps > 0 else 0.0

        # global counters
        self.total_episodes += 1
        self.global_rewards.append(self.ep_reward)
        self.global_lengths.append(self.ep_steps)
        self.global_avg_speeds.append(avg_speed)
        self.global_success.append(self.ep_arrived)
        self.global_collisions.append(self.ep_collided)
        self.global_lane_changes.append(self.ep_lane_changes)
        self.global_truncated.append(self.ep_truncated)
        self.global_teleported.append(self.teleported)
        self.global_removed_unknown.append(self.removed_unknown)
        self.global_offroad.append(self.ep_offroad)
        self.global_action_SS_rate.append(self.ep_actions_SS / self.ep_steps)
        self.global_action_ACC_rate.append(self.ep_actions_ACC / self.ep_steps)
        self.global_action_DEC_rate.append(self.ep_actions_DEC / self.ep_steps)
        self.global_action_LCL_rate.append(self.ep_actions_LCL / self.ep_steps)
        self.global_action_LCR_rate.append(self.ep_actions_LCR / self.ep_steps)

        # snapshot episodio
        self._last_episode_metrics = {
            "ep_reward": self.ep_reward,
            "ep_length": self.ep_steps,
            "ep_avg_speed": avg_speed,
            "ep_collisions": self.ep_collided,
            "ep_offroad": self.ep_offroad,
            "ep_lane_changes": self.ep_lane_changes,
            "ep_arrived": self.ep_arrived,
            "ep_truncated": self.ep_truncated,
            "ep_teleported": self.teleported,
            "ep_removed_unknown": self.removed_unknown,
            "ep_action_SS_rate": self.ep_actions_SS / self.ep_steps,
            "ep_action_ACC_rate": self.ep_actions_ACC / self.ep_steps,
            "ep_action_DEC_rate": self.ep_actions_DEC / self.ep_steps,
            "ep_action_LCL_rate": self.ep_actions_LCL / self.ep_steps,
            "ep_action_LCR_rate": self.ep_actions_LCR / self.ep_steps,
        }

        self.reset()

    # ======================
    # GETTERS
    # ======================
    def get_step_metrics(self):
        return getattr(self, "_last_step_metrics", {})

    def get_episode_metrics(self):
        return getattr(self, "_last_episode_metrics", {})

    def get_rollout_metrics(self):
        if self.total_episodes == 0:
            return {}

        # helper safe
        def safe_std(x):
            return pstdev(x) if len(x) > 1 else 0.0
        
        def slice(self, x):
            return x[-self.window:] if self.window > 0 else x

        return {
            "feat/episodes": self.total_episodes,

            "feat/ep_rew_mean": mean(slice(self, self.global_rewards)),
            "feat/ep_rew_std": safe_std(slice(self, self.global_rewards)),

            "feat/ep_length_mean": mean(slice(self, self.global_lengths)),
            "feat/ep_length_std": safe_std(slice(self, self.global_lengths)),

            "feat/ep_avg_speed_mean": mean(slice(self, self.global_avg_speeds)),
            "feat/ep_avg_speed_std": safe_std(slice(self, self.global_avg_speeds)),

            "end_cause/ep_success_rate": mean(slice(self, self.global_success)),

            "end_cause/ep_collision_rate": mean(slice(self, self.global_collisions)),
            
            "end_cause/ep_offroad_rate": mean(slice(self, self.global_offroad)),

            "end_cause/ep_truncated_rate": mean(slice(self, self.global_truncated)),

            "end_cause/ep_teleported_rate": mean(slice(self, self.global_teleported)),

            "end_cause/ep_removed_unknown_rate": mean(slice(self, self.global_removed_unknown)),

            "action/ep_action_SS_rate": mean(slice(self, self.global_action_SS_rate)),
            "action/ep_action_ACC_rate": mean(slice(self, self.global_action_ACC_rate)),
            "action/ep_action_DEC_rate": mean(slice(self, self.global_action_DEC_rate)),
            "action/ep_action_LCL_rate": mean(slice(self, self.global_action_LCL_rate)),
            "action/ep_action_LCR_rate": mean(slice(self, self.global_action_LCR_rate)),
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