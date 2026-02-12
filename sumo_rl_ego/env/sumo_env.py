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

        ego_exists_before = self.sim.ego_exists(self.ego.id)

        # Caso 1: non è ancora apparso → non applico azione
        if ego_exists_before:
            self.ego.apply_action(self.sim, action)

        self.sim.step()
        self.step_count += 1

        ego_exists_after = self.sim.ego_exists(self.ego.id)

        # Caso 2 o 3: è sparito
        if ego_exists_before and not ego_exists_after:
            obs = self.observation_space.sample()  # dummy obs
            reward = 0.0
            done = True
            return obs, reward, done, False, {"termination_reason": "ego_removed"}

        # Caso 1: non ancora apparso
        if not ego_exists_after:
            obs = self.observation_space.sample()
            return obs, 0.0, False, False, {"termination_reason": "not_spawned"}

        # Caso normale
        obs = self.obs_builder.build(self.sim, self.ego)
        reward = self.reward_fn.compute(self.sim, self.ego)
        done = self.done_fn.check(self.sim, self.ego, self.step_count)

        return obs, reward, done, False, {}


    def close(self):
        self.sim.close()
