import gymnasium as gym
from sumo_rl_ego.env.simulation import SumoSimulation 


class SumoEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.sim = SumoSimulation(config)

        # Strategies
        self.ego = config.ego_class(config.ego_id)
        self.obs_builder = config.obs_class()
        self.reward_fn = config.reward_class()

        self.action_space = self.ego.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.reset()
        self.sim.wait_for_vehicle(self.ego.id)

        # disable SUMO's default lane keeping and speed control for the ego vehicle
        self.sim.enable_rl_control(self.ego.id)

        self.step_count = 0

        # neutral step
        self.sim.step()

        obs = self.obs_builder.build(self.sim, self.ego)
        return obs, {}


    def step(self, action):

        # Inconsistent env: ego missing before action
        if not self.sim.ego_exists(self.ego.id):
            return self._inconsistent_EgoStatus("ego_missing_before_step")

        # Apply action safely
        self._apply_action_safe(action)

        # Advance simulation
        self.sim.step()
        self.step_count += 1

        ego_status = self.sim.get_ego_status(self.ego.id)

        ego_removed_unknown = (
            not ego_status["exists"]
            and not ego_status["collided"]
            and not ego_status["teleported"]
            and not ego_status["arrived"]
        )

        terminated = (
            ego_status["collided"]
            or ego_status["teleported"]
            or ego_status["off_road"]
            or ego_status["arrived"]
            or ego_removed_unknown
        )

        truncated = (
            self.step_count >= self.config.simulation_end and not terminated
        )

        info = dict(ego_status)  
        info.update({
            "ego_removed_unknown": ego_removed_unknown,
            "terminated": terminated,
            "truncated": truncated,
        })

        # Observation
        if terminated or truncated:
            obs = self.observation_space.sample()
        else:
            obs = self.obs_builder.build(self.sim, self.ego)

        # Reward
        if terminated or truncated:
            reward = self.reward_fn.compute_terminal(self.sim, self.ego, info)
        else:
            reward = self.reward_fn.compute(self.sim, self.ego)


        return obs, reward, terminated, truncated, info


    def _inconsistent_EgoStatus(self, reason):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = True
        truncated = False
        info = {"termination_reason": reason}
        return obs, reward, terminated, truncated, info


    def _apply_action_safe(self, action):
        if not self.sim.ego_exists(self.ego.id):
            return False
        try:
            self.ego.apply_action(self.sim, action)
            return True
        except Exception:
            return False


    def close(self):
        self.sim.close()
        print("\nEnvironment closed.")
