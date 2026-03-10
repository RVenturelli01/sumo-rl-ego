from pathlib import Path
from sumo_rl_ego.sumo_envs.registry import register_env
import sumo_gym_ego as sge


BASE_DIR = Path(__file__).resolve().parent.parent / "scenarios" 

# maximum values for clipping and normalization
max_speed = 50.0
max_distance = 200.0
max_ttc = 20.0

# ego parameters
acc_value = 2.0
dec_value = -2.0
lc_duration = 0.0

# reward weights
w_arrived = 1.0
w_crash = -1.0
w_offroad = -0.5
weights = [1.0, 1.0]  # weights for the reward components (speed, terminal)



@register_env("highway_discrete_v1")
class HighwayDiscreteV1(sge.SumoGymEgoEnv):
    def __init__(self, use_gui=False, seed=0):
        sumocfg_files = [str(BASE_DIR / "highway_fast_modified/highway.sumocfg")]

        config = sge.SumoConfig(
            use_gui=use_gui,
            ego_id="ego",
            max_simulation_time=300,
            time_step=0.5,
            seed=seed
        )

        obs_builder = sge.CompositeObservation([
            sge.obs.EgoSpeedObs(max_speed=max_speed),
            sge.obs.EgoLaneObs(),

            sge.obs.NeighborObs(
                neighbors=[
                    "same_front", "same_back",
                    "left_front", "left_back",
                    "right_front", "right_back",
                ],
                features=["distance", "rel_speed", "ttc"],
                max_speed=max_speed,
                max_distance=max_distance,
                max_ttc=max_ttc,
            ),
        ])

        reward_function = sge.CompositeReward([
            sge.reward.SpeedReward(max_speed=max_speed),
            sge.reward.TerminalReward(
                w_crash=w_crash,
                w_offroad=w_offroad,
                w_arrived=w_arrived,)
            ],
            weights=weights
        )

        ego_controller = sge.ego.HighwayDiscreteEgo(
                 acc_value=acc_value,
                 dec_value=dec_value,
                 lc_duration=lc_duration)

        metrics_tracker = sge.CompositeMetricsTracker([
            sge.metrics.EgoFeatureMetrics(),
            sge.metrics.TerminalEventMetrics(),
        ])

        super().__init__(
            sumocfg_files=sumocfg_files,
            config=config,
            obs_builder=obs_builder,
            reward_function=reward_function,
            ego_controller=ego_controller,
            metrics_tracker=metrics_tracker,
        )

