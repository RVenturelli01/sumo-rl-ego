import gymnasium as gym
from sumo_rl_ego.env.simulation import SumoSimulation 
from sumo_rl_ego.ego.base_ego import DefaultEgo
from sumo_rl_ego.observation.base_observation import DefaultObservation
from sumo_rl_ego.reward.base_reward import DefaultReward
from sumo_rl_ego.kpi.base_kpi import DefaultKPI


class SumoEnv(gym.Env):
    def __init__(self, config, ego=None, obs_builder=None, reward_fn=None, kpi_tracker=None):
        super().__init__()

        self.config = config
        self.sim = SumoSimulation(config)

        # --- DEFAULT FALLBACKS ---
        self.ego = ego or DefaultEgo()
        self.obs_builder = obs_builder or DefaultObservation()
        self.reward_fn = reward_fn or DefaultReward()
        self.kpi_tracker = kpi_tracker or DefaultKPI()

        self.ego.setSumoSimulation(self.sim)
        self.obs_builder.setSumoSimulation(self.sim)    
        self.reward_fn.setSumoSimulation(self.sim)
        self.kpi_tracker.setSumoSimulation(self.sim)

        self.ego.setEgoId(config.ego_id)
        self.obs_builder.setEgoId(config.ego_id)    
        self.reward_fn.setEgoId(config.ego_id)
        self.kpi_tracker.setEgoId(config.ego_id)


        self.action_space = self.ego.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.reset()
        self.sim.wait_for_vehicle(self.ego.ego_id)

        # disable SUMO's default lane keeping and speed control for the ego vehicle
        self.sim.enable_rl_control(self.ego.ego_id)

        self.step_count = 0

        # neutral step
        self.sim.simulationStep()

        obs = self.obs_builder.build()
        return obs, {}


    def step(self, action):

        # Inconsistent env: ego missing before action
        if not self.sim.ego_exists(self.ego.ego_id):
            return self._inconsistent_EgoStatus("ego_missing_before_step")

        # Apply action safely
        self._apply_action_safe(action)

        # Advance simulation
        self.sim.simulationStep()
        self.step_count += 1

        ego_status = self.sim.get_ego_status(self.ego.ego_id)


        terminated = (
            ego_status["collided"]
            or ego_status["teleported"]
            or ego_status["off_road"]
            or ego_status["arrived"]
            or ego_status["removed_unknown"]
        )

        truncated = (
            self.step_count >= self.config.simulation_end and not terminated
        )

        ep_status = {
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self.step_count,
            }
        
        # KPI tracking
        kpis_step = self.kpi_tracker.compute_kpis()

        # Reward
        info = {"ego_status": ego_status, "ep_status": ep_status, "kpis_step": kpis_step}
        reward = self.reward_fn.compute(action, info)

        # Observation
        if terminated or truncated:
            obs = self.observation_space.sample()
            kpis_ep = self.kpi_tracker.compute_episode_kpis()
            info["kpis_episode"] = kpis_ep
        else:
            obs = self.obs_builder.build()

        return obs, reward, terminated, truncated, info


    def _inconsistent_EgoStatus(self, reason):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = True
        truncated = False
        info = {"termination_reason": reason}
        return obs, reward, terminated, truncated, info


    def _apply_action_safe(self, action):
        if not self.sim.ego_exists(self.ego.ego_id):
            return False
        try:
            self.ego.apply_action(action)
            return True
        except Exception:
            return False


    def close(self):
        self.sim.close()
        print("\nEnvironment closed.")
