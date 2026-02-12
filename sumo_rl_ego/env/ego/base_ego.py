from abc import ABC, abstractmethod

class BaseEgoVehicle(ABC):

    def __init__(self, veh_id):
        self.id = veh_id

    @abstractmethod
    def apply_action(self, sim, action):
        pass
