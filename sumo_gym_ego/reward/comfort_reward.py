from sumo_gym_ego import BaseRewardFunction



class ComfortReward(BaseRewardFunction):

    def __init__(self, max_acc=2.0, max_dec=-2.0, w_acc=1.0, w_jerk=1.0):
        self.max_acc = max(max_acc, -max_dec)

        self.w_acc = w_acc
        self.w_jerk = w_jerk

        self.v_prev = None
        self.a_prev = None

    def compute(self, obs, action, next_obs, info):
        time_step = self.config.time_step

        v = self.sim.vehicle.getSpeed(self.ego_id)

        reward = 0.0

        if self.v_prev is not None:
            a = (v - self.v_prev) / time_step
            reward -= self.w_acc * abs(a) / self.max_acc

        if self.a_prev is not None:
            jerk = (a - self.a_prev) / (2 * self.max_acc)  
            reward -= self.w_jerk * abs(jerk) 

        if self.v_prev is not None:
            self.a_prev = a
            
        self.v_prev = v

        return reward

    def compute_terminal(self, obs, action, next_obs, info):
        return 0.0