from abc import ABC, abstractmethod


class BaseReward(ABC):

    @abstractmethod
    def compute(self, sim, ego):
        """
        Compute step reward when episode is ongoing.
        """
        pass

    @abstractmethod
    def compute_terminal(self, sim, info: dict) -> float:
        """
        Compute reward when episode terminates or is truncated.
        """
        pass
