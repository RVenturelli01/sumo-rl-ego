from abc import ABC, abstractmethod


class Policy(ABC):

    def reset(self):
        """Called at episode reset (optional)"""
        pass

    @abstractmethod
    def predict(self, obs):
        """Return action"""
        pass
