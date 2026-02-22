import gymnasium as gym
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.simulation import SumoSimulation 
from sumo_rl_ego.ego.base import DefaultEgoController
from sumo_rl_ego.observation.base import DefaultObservationBuilder
from sumo_rl_ego.reward.base import DefaultRewardFunction
from sumo_rl_ego.metrics.base import DefaultMetricsTracker


class SumoEnv(gym.Env):
    def __init__(self, 
                 config = None, 
                 ego_controller=None, 
                 obs_builder=None, 
                 reward_function=None, 
                 metrics_tracker=None):
        
        super().__init__()


        # --- DEFAULT FALLBACKS ---
        self.config = config or SumoConfig()
        self.ego_controller = ego_controller or DefaultEgoController()
        self.obs_builder = obs_builder or DefaultObservationBuilder()
        self.reward_function = reward_function or DefaultRewardFunction()
        self.metrics_tracker = metrics_tracker or DefaultMetricsTracker()
        self.sim = SumoSimulation(self.config)

        self.ego_controller.set_sumo_simulation(self.sim)
        self.obs_builder.set_sumo_simulation(self.sim)    
        self.reward_function.set_sumo_simulation(self.sim)
        self.metrics_tracker.set_sumo_simulation(self.sim)

        self.ego_controller.set_ego_id(self.config.ego_id)
        self.obs_builder.set_ego_id(self.config.ego_id)    
        self.reward_function.set_ego_id(self.config.ego_id)
        self.metrics_tracker.set_ego_id(self.config.ego_id)


        self.action_space = self.ego_controller.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0


    def reset(self, seed=None, options=None):

        # Call super to set the seed properly for Gymnasium compatibility
        super().reset(seed=seed) 

        # ensure different seed at each reset for more varied episodes
        self.config.seed += 1  
        
        self.sim.reset()
        self.ego_controller.reset()
        self.obs_builder.reset()
        self.reward_function.reset()
        self.metrics_tracker.reset()

        # Wait for ego vehicle to be loaded in the simulation before proceeding
        self.sim.wait_for_vehicle(self.ego_controller.ego_id)

        # disable SUMO's default lane keeping and speed control for the ego vehicle
        self.sim.enable_rl_control(self.ego_controller.ego_id)

        self.step_count = 0

        # neutral step
        self.sim.simulationStep()

        obs = self.obs_builder.build_obs()
        return obs, {}


    def step(self, action):

        # Inconsistent env: ego missing before action
        if not self.sim.ego_exists(self.ego_controller.ego_id):
            print(f"Warning: Ego vehicle is missing before step") 
            return self._inconsistent_EgoStatus("ego_missing_before_step")

        self.ego_controller.apply_action(action)

        # Advance simulation
        self.sim.simulationStep()
        self.step_count += 1

        # ego_status: collided, teleported, off_road, arrived, removed_unknown
        ego_status = self.sim.get_ego_status(self.ego_controller.ego_id)

        terminated = (
            ego_status["collided"]
            or ego_status["teleported"]
            or ego_status["off_road"]
            or ego_status["arrived"]
            or ego_status["removed_unknown"]
        )

        truncated = (
            self.step_count >= self.config.max_steps and not terminated
        )

        ep_status = {
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self.step_count,
            }
        
        # Info dict
        info = {"ego_status": ego_status, "ep_status": ep_status}
        
        # Observation
        if terminated or truncated:
            obs = self.observation_space.sample()
        else:
            obs = self.obs_builder.build_obs()

        # Reward
        reward = self.reward_function.compute(action, info)

        # Metrics 
        self.metrics_tracker.update_step(obs, action, reward, info)
        info["metrics"] = self.metrics_tracker.get_step_metrics()

        if terminated or truncated:
            self.metrics_tracker.end_episode(info)
            info["episode_metrics"] = self.metrics_tracker.get_episode_metrics()
            info["rollout_metrics"] = self.metrics_tracker.get_rollout_metrics()

        return obs, reward, terminated, truncated, info


    def _inconsistent_EgoStatus(self, reason):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = True
        truncated = False
        info = {"termination_reason": reason}
        return obs, reward, terminated, truncated, info


    def close(self):
        self.sim.close()
        print("\nEnvironment closed.")
