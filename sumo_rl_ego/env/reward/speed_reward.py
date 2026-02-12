from sumo_rl_ego.env.reward.base_reward import BaseReward

class SpeedReward(BaseReward):

    def compute(self, sim, ego):
        speed = sim.get_speed(ego.id)
        return speed / 30.0  # normalizzato
