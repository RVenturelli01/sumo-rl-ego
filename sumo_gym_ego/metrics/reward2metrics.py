from sumo_gym_ego import BaseMetricsTracker, BaseRewardFunction

class Reward2Metrics(BaseMetricsTracker):

    def __init__(self, reward_function: BaseRewardFunction, reward_name: str = "???"):
        self.reward_function = reward_function
        self.reward_name = reward_name
        self.reward_sum = 0.0

        self.reset()


    def reset(self):
        self.reward_sum = 0.0
        self.reward_function.reset()


    def bind(self, config, sim, sim_bus):
        self.reward_function.bind(config, sim, sim_bus)


    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        self.reward_sum += self.reward_function.compute(obs, action, next_obs, info)

        return {}


    def compute_episode_metrics(self, obs, action, next_obs, reward, info):
        self.reward_sum += self.reward_function.compute_terminal(obs, action, next_obs, info)

        return {
            f"rewards/{self.reward_name}": self.reward_sum,
        }

