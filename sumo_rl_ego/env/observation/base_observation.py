from abc import ABC, abstractmethod

class BaseObservation(ABC):

    @abstractmethod
    def build(self, sim, ego):
        pass
