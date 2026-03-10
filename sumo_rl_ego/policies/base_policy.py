from abc import ABC, abstractmethod

class BasePolicy(ABC):

    def reset(self):
        """Called at episode reset (optional)"""
        pass

    @abstractmethod
    def predict(self, obs):
        """Return action"""
        pass