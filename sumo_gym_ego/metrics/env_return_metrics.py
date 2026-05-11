from sumo_gym_ego import BaseMetricsTracker, BaseRewardFunction

class EnvReturnMetrics(BaseMetricsTracker):

    def __init__(self):
        self.reward_sum = 0.0

        self.reset()


    def reset(self):
        self.reward_sum = 0.0


    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        self.reward_sum += reward

        return {}


    def compute_episode_metrics(self, obs, action, next_obs, reward, info):
        self.reward_sum += reward

        return {
            f"rewards/ep_env_return": self.reward_sum,
        }

