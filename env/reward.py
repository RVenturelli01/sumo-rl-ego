class RewardFunction:
    def __init__(self, target_speed=15.0):
        self.target_speed = target_speed

    def compute(self, sim, ego):
        state = sim.get_vehicle_state(ego.id)

        speed_error = abs(state["speed"] - self.target_speed)

        reward = -speed_error

        if sim.has_collision():
            reward -= 100.0

        return reward
