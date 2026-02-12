class SpeedReward:

    def compute(self, sim, ego):
        speed = sim.get_speed(ego.id)
        return speed / 30.0  # normalizzato
