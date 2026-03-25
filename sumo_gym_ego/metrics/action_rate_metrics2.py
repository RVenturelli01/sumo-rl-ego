from sumo_gym_ego import BaseMetricsTracker
from sumo_gym_ego.ego.highway_discrete_ego import DiscreteActions


class ActionRateMetrics2(BaseMetricsTracker):

    def __init__(
            self,
            max_acc=2.0,        # m/s^2
            max_dec=-2.0,       # m/s^2
            lane_threshold=0.5  # threshold for lane change
        ):
        
        self.max_acc = max_acc
        self.max_dec = max_dec
        self.lane_threshold = lane_threshold
        self.step_count = 0

        self.reset()

    def reset(self):
        self.count_ss = 0
        self.count_acc = 0
        self.count_dec = 0
        self.count_lcl = 0
        self.count_lcr = 0
        self.step_count = 0


    def compute_step_metrics(self, obs, action, next_obs, reward, info):
        a_long = action[0]
        lane_cmd = action[1]

        if a_long > self.max_acc*0.2:
            self.count_acc += 1
        elif a_long < self.max_dec*0.2:
            self.count_dec += 1
        else:
            self.count_ss += 1
        
        if lane_cmd > self.lane_threshold:
            self.count_lcl += 1
        elif lane_cmd < -self.lane_threshold:
            self.count_lcr += 1

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
