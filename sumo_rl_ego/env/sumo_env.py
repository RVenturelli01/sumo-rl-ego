import gymnasium as gym
from sumo_rl_ego.env.simulation import SumoSimulation


class SumoEnv(gym.Env):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sim = SumoSimulation(config)

        # Instantiate strategies
        self.ego = config.ego_class(config.ego_id)
        self.obs_builder = config.obs_class()
        self.reward_fn = config.reward_class()
        self.done_fn = config.done_class()

        self.action_space = self.ego.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.sim.start()
        self.step_count = 0

        obs = self.obs_builder.build(self.sim, self.ego)
        return obs, {}

    def step(self, action):
        self.ego.apply_action(self.sim, action)
        self.sim.step()

        self.step_count += 1

        obs = self.obs_builder.build(self.sim, self.ego)
        reward = self.reward_fn.compute(self.sim, self.ego)
        done = self.done_fn.check(self.sim, self.ego, self.step_count)

        return obs, reward, done, False, {}

    def close(self):
        self.sim.close()
