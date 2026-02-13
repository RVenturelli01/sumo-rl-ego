from abc import ABC, abstractmethod


class BaseReward(ABC):

    @abstractmethod
    def compute(self, sim, ego):
        """
        Compute step reward when episode is ongoing.
        """
        pass

    @abstractmethod
    def compute_terminal(
        self,
        has_collided: bool,
        has_teleported: bool,
        is_off_road: bool,
        route_completed: bool,
        ego_removed_unknown: bool,
        truncated_due_to_timeout: bool,
    ):
        """
        Compute reward when episode terminates or is truncated.
        """
        pass
