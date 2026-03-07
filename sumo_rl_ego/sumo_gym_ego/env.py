import random
import warnings
import gymnasium as gym

from sumo_rl_ego.sumo_gym_ego.core.config import SumoConfig
from sumo_rl_ego.sumo_gym_ego.core.ego_status import EgoStatus
from sumo_rl_ego.sumo_gym_ego.core.sim_bus import SimBus
from sumo_rl_ego.sumo_gym_ego.core.simulation import SumoSimulation
from sumo_rl_ego.sumo_gym_ego.ego.default import DefaultEgoController
from sumo_rl_ego.sumo_gym_ego.observation.default import DefaultObservationBuilder
from sumo_rl_ego.sumo_gym_ego.reward.default import DefaultRewardFunction
from sumo_rl_ego.sumo_gym_ego.metrics.default import DefaultMetricsTracker


class SumoEnv(gym.Env):
    def __init__(self,
                 sumocfg_files: list[str],
                 config=None,
                 ego_controller=None,
                 obs_builder=None,
                 reward_function=None,
                 metrics_tracker=None):

        super().__init__()

        self.sumocfg_files = sumocfg_files
        self.config = config or SumoConfig()
        self.ego_controller = ego_controller or DefaultEgoController()
        self.obs_builder = obs_builder or DefaultObservationBuilder()
        self.reward_function = reward_function or DefaultRewardFunction()
        self.metrics_tracker = metrics_tracker or DefaultMetricsTracker()

        self.sim = SumoSimulation(self.config)
        self.ego_id = self.config.ego_id

        sim_bus = SimBus()
        self.ego_controller.bind(self.config, self.sim, sim_bus)
        self.obs_builder.bind(self.config, self.sim, sim_bus)
        self.reward_function.bind(self.config, self.sim, sim_bus)
        self.metrics_tracker.bind(self.config, self.sim, sim_bus)

        self.action_space = self.ego_controller.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0
        self.last_obs = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # different seed at each reset for more varied episodes
        self.config.seed += 1
        self.config.sumocfg_file = random.choice(self.sumocfg_files)

        self.sim.reset()
        self.ego_controller.reset()
        self.obs_builder.reset()
        self.reward_function.reset()
        self.metrics_tracker.reset()

        # wait for ego vehicle to appear in simulation
        self.sim.wait_for_vehicle(self.ego_id)

        # disable SUMO's default lane keeping and speed control
        self.sim.enable_rl_control(self.ego_id)

        self.step_count = 0

        # neutral step
        self.sim.simulationStep()

        obs = self.obs_builder.build_obs()
        self.last_obs = obs

        return obs, {}

    def step(self, action):
        
        # inconsistent env: ego missing before action
        if not self.sim.ego_exists(self.ego_id):
            warnings.warn(
                "Ego vehicle missing before step. Did you forget to call reset()?",
                RuntimeWarning,)
            info = {"termination_reason": "ego_missing_before_step"}
            return self.last_obs, 0.0, True, False, info

        self.ego_controller.apply_action(action)

        self.sim.simulationStep()
        self.step_count += 1

        ego_status = self.sim.get_ego_status(self.ego_id)

        terminated, truncated, event = self._compute_termination(ego_status)

        info = {
            "status": {
                "step": self.step_count,
                "terminated": terminated,
                "truncated": truncated,
            },
            "event": event,
        }

        if terminated or truncated:
            obs = self.last_obs
        else:
            obs = self.obs_builder.build_obs()

        reward = self.reward_function.compute(self.last_obs, action, obs, info)

        info["metrics"] = {}
        info["metrics"]["step"] = self.metrics_tracker.compute_step_metrics(self.last_obs, action, obs, reward, info)

        if terminated or truncated:
            info["metrics"]["episode"] = self.metrics_tracker.finalize_episode_metrics(info)
            info["log"] = self.metrics_tracker.get_log_metrics()

        self.last_obs = obs
        return obs, reward, terminated, truncated, info


    def _compute_termination(self, ego_status):
        terminated = ego_status != EgoStatus.RUNNING
        truncated = self.step_count >= self.config.max_steps and not terminated

        if terminated:
            event = ego_status.value
        elif truncated:
            event = "timeout"
        else:
            event = None

        return terminated, truncated, event


    def close(self):
        self.sim.close()
        print("\nEnvironment closed.")
