from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, env=None):
        self.env = env

    def reset(self):
        """Called at episode reset (optional)"""
        pass

    @abstractmethod
    def predict(self, obs):
        """Return action"""
        pass
