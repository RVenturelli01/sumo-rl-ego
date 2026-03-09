from abc import ABC


class BaseEnvPlugin(ABC):
    """Common base for all environment plugins (ego, observation, reward, metrics)."""

    def reset(self):
        pass

    def bind(self, config, sim, sim_bus):
        self.config = config
        self.sim = sim
        self.ego_id = config.ego_id
        self.sim_bus = sim_bus
    
