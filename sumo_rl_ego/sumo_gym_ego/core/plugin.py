from abc import ABC, abstractmethod


class BaseEnvPlugin(ABC):
    """Common base for all environment plugins (ego, observation, reward, metrics)."""

    def reset(self):
        pass

    def set_context(self, ctx):
        self.ctx = ctx

    @property
    def sim(self):
        return self.ctx.sim

    @property
    def ego_id(self):
        return self.ctx.config.ego_id
