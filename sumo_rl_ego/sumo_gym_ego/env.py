import random
import gymnasium as gym

from sumo_rl_ego.sumo_gym_ego.core.config import SumoConfig
from sumo_rl_ego.sumo_gym_ego.core.context import EnvContext
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

        ctx = EnvContext(self.config, self.sim)
        self.ego_controller.set_context(ctx)
        self.obs_builder.set_context(ctx)
        self.reward_function.set_context(ctx)
        self.metrics_tracker.set_context(ctx)

        self.action_space = self.ego_controller.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0

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
        self.sim.wait_for_vehicle(self.ego_controller.ego_id)

        # disable SUMO's default lane keeping and speed control
        self.sim.enable_rl_control(self.ego_controller.ego_id)

        self.step_count = 0

        # neutral step
        self.sim.simulationStep()

        obs = self.obs_builder.build_obs()
        return obs, {}

    def step(self, action):
        # inconsistent env: ego missing before action
        if not self.sim.ego_exists(self.ego_controller.ego_id):
            print("Warning: Ego vehicle is missing before step")
            return self._inconsistent_ego_status("ego_missing_before_step")

        self.ego_controller.apply_action(action)

        self.sim.simulationStep()
        self.step_count += 1

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

        info = {"ego_status": ego_status, "ep_status": ep_status}

        if terminated or truncated:
            obs = self.observation_space.sample()
        else:
            obs = self.obs_builder.build_obs()

        reward = self.reward_function.compute(action, info)

        info["metrics"] = {}
        info["metrics"]["step"] = self.metrics_tracker.compute_step_metrics(obs, action, reward, info)

        if terminated or truncated:
            info["metrics"]["episode"] = self.metrics_tracker.finalize_episode_metrics()
            info["log"] = self.metrics_tracker.get_log_metrics()

        return obs, reward, terminated, truncated, info

    def _inconsistent_ego_status(self, reason):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = True
        truncated = False
        info = {"inconsistent_ego_status": reason}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.sim.close()
        print("\nEnvironment closed.")
