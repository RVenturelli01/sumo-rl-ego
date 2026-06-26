from sumo_gym_ego import BaseMetricsTracker
from sumo_gym_ego.ego.highway_discrete_ego import DiscreteActions


class DiscreteActionRateMetrics(BaseMetricsTracker):

    def __init__(self):
        self.reset()

    def reset(self):
        self.count_ss = 0
        self.count_acc = 0
        self.count_dec = 0
        self.count_lcl = 0
        self.count_lcr = 0
        self.step_count = 0


    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        self.count_ss += action == DiscreteActions.SS
        self.count_acc += action == DiscreteActions.ACC
        self.count_dec += action == DiscreteActions.DEC
        self.count_lcl += action == DiscreteActions.LCL
        self.count_lcr += action == DiscreteActions.LCR

        self.step_count += 1
        return {}

    def compute_episode_metrics(self, obs, action, next_obs, reward, info):

        step_count = self.step_count

        return {
            "action_rate/ss": self.count_ss / step_count if step_count > 0 else 0.0,
            "action_rate/acc": self.count_acc / step_count if step_count > 0 else 0.0,
            "action_rate/dec": self.count_dec / step_count if step_count > 0 else 0.0,
            "action_rate/lcl": self.count_lcl / step_count if step_count > 0 else 0.0,
            "action_rate/lcr": self.count_lcr / step_count if step_count > 0 else 0.0,
        }
