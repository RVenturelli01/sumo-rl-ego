from pathlib import Path
from sumo_rl_ego.sumo_envs.registry import register_env
import sumo_gym_ego as sge
from typing import Literal
from dataclasses import dataclass

'''
Scenario description:

the environment simulates a highway scenario in sumo, the highway has three lanes and has a lot of traffic
the ego vehicle must reach the end of the highway without crashing, going offroad, or timing out.

the ego at each iteration obtains an observation that includes:
- its own speed 
- the lane it is currently in
- whether the lanes to the left and right are free for a lane change
- the distance to the nearest vehicle in front and behind in the same lane
- the distance to the nearest vehicle in front and behind in the left lane (if it exists)
- the distance to the nearest vehicle in front and behind in the right lane (if it exists)

the ego can either have:
- discrete action space (acc, dec, lane change left, lane change right, no-op) 
- continuous action space (acceleration and lane change commands).

the reward function can be configured to prioritize different aspects of the task
- speed
- safety
- comfort 

'''


BASE_DIR = Path(__file__).resolve().parent.parent / "scenarios" 


@dataclass
class ENV_PARAMS:
    time_step: float = 0.5
    max_simulation_time: int = 300
    num_lanes: int = 3

    max_speed: float = 50.0
    max_distance: float = 200.0
    max_ttc: float = 20.0
    check_distance: float = 10.0

    acc_value: float = 2.0
    dec_value: float = -2.0
    lc_duration: float = 0.0
    max_acc: float = 2.0
    max_dec: float = -2.0
    lane_threshold: float = 0.9

    w_penalty: float = -0.2
    w_speed: float = 1.0
    w_arrived: float = 0.0
    w_crash: float = -10.0
    w_offroad: float = -10.0
    w_timeout: float = -10.0

    

@register_env("HighwayEgo-v0")
class HighwayEgo_v0(sge.SumoEnv):
    def __init__(self, 
                 reward: Literal["fast", "safe", "comfort"] = "fast",
                 ego: Literal["discrete", "continuous"] = "discrete",
                 metrics_tracker = None,
                 use_gui=False, 
                 seed=0,
                 ):
        
        sumocfg_files = [str(BASE_DIR / "highway_fast_modified/highway.sumocfg")]

        config = sge.SumoConfig(
            use_gui=use_gui,
            ego_id="ego",
            max_simulation_time=ENV_PARAMS.max_simulation_time,
            time_step=ENV_PARAMS.time_step,
            seed=seed
        )

        obs_builder = sge.CompositeObservation([
            sge.obs.EgoSpeedObs(max_speed=ENV_PARAMS.max_speed),
            sge.obs.EgoLaneObs(),
            sge.obs.LaneFreeObs(check_distance=ENV_PARAMS.check_distance),
            sge.obs.NeighborObs(
                neighbors=[
                    "same_front", "same_back",
                    "left_front", "left_back",
                    "right_front", "right_back",
                ],
                features=["distance", "rel_speed"],
                max_speed=ENV_PARAMS.max_speed,
                max_distance=ENV_PARAMS.max_distance,
                max_ttc=ENV_PARAMS.max_ttc,
            ),
        ])

        if reward == "fast":
            reward_function = sge.CompositeReward([
                sge.reward.StepPenalty(penalty=ENV_PARAMS.w_penalty),
                sge.reward.SpeedReward(max_speed=ENV_PARAMS.max_speed, weight=ENV_PARAMS.w_speed),
                sge.reward.TerminalReward(
                    w_crash=ENV_PARAMS.w_crash,
                    w_offroad=ENV_PARAMS.w_offroad,
                    w_arrived=ENV_PARAMS.w_arrived,
                    w_timeout=ENV_PARAMS.w_timeout,)
                ]
            )

        if ego == "discrete":
            ego_controller = sge.ego.HighwayDiscreteEgo(
                    acc_value=ENV_PARAMS.acc_value,
                    dec_value=ENV_PARAMS.dec_value,
                    lc_duration=ENV_PARAMS.lc_duration)
        elif ego == "continuous":
            ego_controller = sge.ego.HighwayContinuousEgo(
                max_acc=ENV_PARAMS.max_acc,
                max_dec=ENV_PARAMS.max_dec,
                lc_duration=ENV_PARAMS.lc_duration,
                lane_threshold=ENV_PARAMS.lane_threshold,
            )

        if metrics_tracker is None:
            if ego == "discrete":
                metrics_tracker = sge.CompositeMetricsTracker([
                    sge.metrics.PerformanceMetrics(),
                    sge.metrics.ActionRateMetrics(),
                ])
            elif ego == "continuous":
                metrics_tracker = sge.CompositeMetricsTracker([
                    sge.metrics.PerformanceMetrics(),
                    sge.metrics.ActionRateMetrics2(
                        max_acc=ENV_PARAMS.max_acc,
                        max_dec=ENV_PARAMS.max_dec,
                        lane_threshold=ENV_PARAMS.lane_threshold,
                    ),
                ])

        super().__init__(
            sumocfg_files=sumocfg_files,
            config=config,
            obs_builder=obs_builder,
            reward_function=reward_function,
            ego_controller=ego_controller,
            metrics_tracker=metrics_tracker,
        )
