from abc import ABC, abstractmethod

class BaseReward(ABC):

    @abstractmethod
    def compute(self, sim, ego):
        pass
