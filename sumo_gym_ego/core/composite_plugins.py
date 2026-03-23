from gymnasium.spaces import Box
import numpy as np

from .base_plugins import ( 
    BaseEnvPlugin, 
    BaseRewardFunction, 
    BaseMetricsTracker, 
    BaseObservationBuilder
)




class CompositePlugin(BaseEnvPlugin):

    def __init__(self, plugins):
        self.plugins = plugins

    # --------------------------------
    # lifecycle
    # --------------------------------

    def bind(self, config, sim, sim_bus):

        super().bind(config, sim, sim_bus)

        for p in self.plugins:
            p.bind(config, sim, sim_bus)

    def reset(self):

        for p in self.plugins:
            p.reset()






class CompositeReward(CompositePlugin, BaseRewardFunction):

    def __init__(self, rewards, weights=None):

        super().__init__(rewards)

        if weights is None:
            weights = [1.0] * len(rewards)

        self.weights = weights

    def compute(self, obs, action, next_obs, info):

        total = 0.0

        for r, w in zip(self.plugins, self.weights):
            total += w * r.compute(obs, action, next_obs, info)

        return total

    def compute_terminal(self, obs, action, next_obs, info):

        total = 0.0

        for r, w in zip(self.plugins, self.weights):
            total += w * r.compute_terminal(obs, action, next_obs, info)

        return total
    




class CompositeMetricsTracker(CompositePlugin, BaseMetricsTracker):

    def compute_step_metrics(self, obs, action, next_obs, reward, info):

        metrics = {}

        for m in self.plugins:
            metrics.update(m.compute_step_metrics(obs, action, next_obs, reward, info))

        return metrics

    def compute_episode_metrics(self, obs, action, next_obs, reward, info):

        metrics = {}

        for m in self.plugins:
            metrics.update(m.compute_episode_metrics(obs, action, next_obs, reward, info))

        return metrics



    

class CompositeObservation(CompositePlugin, BaseObservationBuilder):

    def __init__(self, observations):

        super().__init__(observations)

        lows = []
        highs = []

        for o in observations:
            lows.append(o.observation_space.low)
            highs.append(o.observation_space.high)

        self.observation_space = Box(
            low=np.concatenate(lows),
            high=np.concatenate(highs),
            dtype=np.float64,
        )

    def build_obs(self):

        obs = []

        for o in self.plugins:
            obs.append(o.build_obs())

        return np.concatenate(obs)
    
    
    def print_obs(self, obs):

        idx = 0

        for o in self.plugins:

            size = o.observation_space.shape[0]
            obs_part = obs[idx: idx + size]

            if hasattr(o, "print_obs"):
                o.print_obs(obs_part)

            idx += size