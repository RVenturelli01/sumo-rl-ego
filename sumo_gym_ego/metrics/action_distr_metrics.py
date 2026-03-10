from statistics import mean
from types import SimpleNamespace
from sumo_gym_ego import BaseMetricsTracker
from sumo_gym_ego.ego.highway_discrete_ego import DiscreteActions


class ActionDistrMetrics(BaseMetricsTracker):

    def __init__(self, window=100):

        self.window = window

        self.episode = SimpleNamespace()
        self.history = SimpleNamespace()

        self.reset_global()
        self.reset()

    def reset(self):

        ep = self.episode

        ep.ss = 0
        ep.acc = 0
        ep.dec = 0
        ep.lcl = 0
        ep.lcr = 0

    def reset_global(self):

        self.history.ss = []
        self.history.acc = []
        self.history.dec = []
        self.history.lcl = []
        self.history.lcr = []

    def compute_step_metrics(self, obs, action, next_obs, reward, info):

        ep = self.episode

        ep.ss += action == DiscreteActions.SS
        ep.acc += action == DiscreteActions.ACC
        ep.dec += action == DiscreteActions.DEC
        ep.lcl += action == DiscreteActions.LCL
        ep.lcr += action == DiscreteActions.LCR

        return {}

    def finalize_episode_metrics(self, info):

        ep = self.episode
        hist = self.history

        step_count = info.get("status", {}).get("step", 0)

        hist.ss.append(self._safe_div(ep.ss, step_count))
        hist.acc.append(self._safe_div(ep.acc, step_count))
        hist.dec.append(self._safe_div(ep.dec, step_count))
        hist.lcl.append(self._safe_div(ep.lcl, step_count))
        hist.lcr.append(self._safe_div(ep.lcr, step_count))

        return {
            "ep_ss_rate": hist.ss[-1],
            "ep_acc_rate": hist.acc[-1],
            "ep_dec_rate": hist.dec[-1],
            "ep_lcl_rate": hist.lcl[-1],
            "ep_lcr_rate": hist.lcr[-1],
        }

    def get_log_metrics(self):

        hist = self.history

        def win(x):
            return x[-self.window:] if self.window > 0 else x

        return {
            "action/ep_ss_rate": mean(win(hist.ss)),
            "action/ep_acc_rate": mean(win(hist.acc)),
            "action/ep_dec_rate": mean(win(hist.dec)),
            "action/ep_lcl_rate": mean(win(hist.lcl)),
            "action/ep_lcr_rate": mean(win(hist.lcr)),
        }

    @staticmethod
    def _safe_div(a, b):
        return a / b if b > 0 else 0.0