import gymnasium as gym
import traci


class SumoEnv(gym.Env):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Strategies
        self.ego = config.ego_class(config.ego_id)
        self.obs_builder = config.obs_class()
        self.reward_fn = config.reward_class()
        self.termination_fn = config.termination_class()

        self.action_space = self.ego.action_space
        self.observation_space = self.obs_builder.observation_space

        self.step_count = 0
        self.started = False


    def reset(self, seed=None, options=None):

        # Start / restart simulation
        if self.started:
            traci.load(self.config.build_cmd()[1:])
        else:
            traci.start(self.config.build_cmd())
            self.started = True

        # Wait until ego vehicle is spawned
        max_wait_steps = 10000
        wait = 0
        while self.ego.id not in traci.vehicle.getIDList():
            traci.simulationStep()
            wait += 1
            if wait > max_wait_steps:
                raise RuntimeError("Ego vehicle was not spawned in simulation.")

        # Enable RL control / disable default SUMO behavior for ego
        traci.vehicle.setSpeedMode(self.ego.id, 7)
        traci.vehicle.setLaneChangeMode(self.ego.id, 0)

        self.step_count = 0

        # Neutral step
        traci.simulationStep()

        obs = self.obs_builder.build(traci, self.ego)

        return obs, {}


    def step(self, action):

        # Ego missing before applying action → environment inconsistent
        if not self.ego.id in traci.vehicle.getIDList():
            obs = self.observation_space.sample()
            reward = 0.0
            terminated = True
            truncated = False
            info = {"termination_reason": "ego_missing_before_step"}

            return obs, reward, terminated, truncated, info

        # Apply action
        self.ego.apply_action(traci, action)

        traci.simulationStep()
        self.step_count += 1

        ego_exists = self.ego.id in traci.vehicle.getIDList()
        has_collided = self.ego.id in traci.simulation.getCollidingVehiclesIDList()
        has_teleported = self.ego.id in traci.simulation.getStartingTeleportIDList()
        is_off_road = ego_exists and traci.vehicle.getLaneID(self.ego.id) == ""
        time_out = self.step_count >= self.config.simulation_end
        route_completed = self.ego.id in traci.simulation.getArrivedIDList()
        ego_removed_unknown = not ego_exists and not has_collided and not has_teleported and not is_off_road and not route_completed

        terminated = has_collided or has_teleported or is_off_road or route_completed or ego_removed_unknown
        truncated = time_out and not terminated
        # normal = not (terminated or truncated)

        # Observation
        if terminated or truncated:
            obs = self.observation_space.sample()  # observation neutra
        else:
            obs = self.obs_builder.build(traci, self.ego)

        # Reward
        if terminated or truncated:
            reward = self.reward_fn.compute_terminal(
                has_collided=has_collided,
                has_teleported=has_teleported,
                is_off_road=is_off_road,
                route_completed=route_completed,
                ego_removed_unknown=ego_removed_unknown,
                truncated_due_to_timeout=time_out
            )

        else:
            reward = self.reward_fn.compute(traci, self.ego)

        # Info dictionary
        if terminated or truncated:
            if has_collided:
                termination_reason = "collision"
            elif has_teleported:
                termination_reason = "teleportation"
            elif is_off_road:
                termination_reason = "off_road"
            elif route_completed:
                termination_reason = "arrived"
            elif ego_removed_unknown:
                termination_reason = "unknown_removal"
            elif time_out:
                termination_reason = "time_out"
            else:
                termination_reason = "other"
        else:
            termination_reason = ""
            
        info = {"termination_reason": termination_reason}

        return obs, reward, terminated, truncated, info




    def close(self):
        if self.started:
            traci.close()
            self.started = False
