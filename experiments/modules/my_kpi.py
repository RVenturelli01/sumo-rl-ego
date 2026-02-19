from sumo_rl_ego.kpi.base_kpi import BaseKPI


class MyKPI(BaseKPI):

    def __init__(self):
        self.ego_speed_history = []


    def compute_kpi(self):
        self.left_front = self.sim.vehicle.getNeighbors(self.ego_id, 0b010)
        self.left_back = self.sim.vehicle.getNeighbors(self.ego_id, 0b000)
        self.right_front = self.sim.vehicle.getNeighbors(self.ego_id, 0b011)
        self.right_back = self.sim.vehicle.getNeighbors(self.ego_id, 0b001)
        
        self.left_front_blocking = self.sim.vehicle.getLeftLeaders(self.ego_id, True)
        self.left_back_blocking = self.sim.vehicle.getLeftFollowers(self.ego_id, True)
        self.right_front_blocking = self.sim.vehicle.getRightLeaders(self.ego_id, True)
        self.right_back_blocking = self.sim.vehicle.getRightFollowers(self.ego_id, True)

        self.ego_speed_history.append(self.sim.vehicle.getSpeed(self.ego_id))

        return {
            "left_front": self.left_front,
            "left_back": self.left_back,
            "right_front": self.right_front,
            "right_back": self.right_back,
            "left_front_blocking": self.left_front_blocking,
            "left_back_blocking": self.left_back_blocking,
            "right_front_blocking": self.right_front_blocking,
            "right_back_blocking": self.right_back_blocking
        }

    def compute_episode_kpis(self):
        if self.ego_speed_history:
            return {
                "avg_speed": sum(self.ego_speed_history) / len(self.ego_speed_history),
                "max_speed": max(self.ego_speed_history),
                "min_speed": min(self.ego_speed_history)
            }
        return {}