from pathlib import Path
from sumo_rl_ego.sumo_envs.registry import register_env
import sumo_gym_ego as sge
from typing import Literal

'''
Scenario description:

the environment simulates a highway scenario in sumo, the highway has three lanes and has a lot of traffic
the ego vehicle must reach the end of the highway without crashing, going offroad, or timing out.

the ego at each iteration obtains an observation that includes:
- its own speed and lane
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


# simulation parameters
time_step = 0.5
max_simulation_time = 300

# maximum values for clipping and normalization
max_speed = 50.0
max_distance = 200.0
max_ttc = 20.0
check_distance = 10.0

# ego parameters
acc_value = 2.0    # discrete ego
dec_value = -2.0    # discrete ego
lc_duration = 0.0   # both discrete and continuous ego
max_acc = 2.0   # continuous ego
max_dec = -2.0   # continuous ego
lane_threshold = 0.9   # continuous ego

# reward weights
step_penalty = -0.2
w_arrived = 0.0
w_crash = -10.0
w_offroad = -10.0
w_timeout = -10.0
weights = [1.0, 1.0, 1.0]  # weights for the reward components (step_penalty, speed, terminal)



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
            max_simulation_time=max_simulation_time,
            time_step=time_step,
            seed=seed
        )

        obs_builder = sge.CompositeObservation([
            sge.obs.EgoSpeedObs(max_speed=max_speed),
            sge.obs.EgoLaneObs(),
            sge.obs.LaneFreeObs(check_distance=check_distance),
            sge.obs.NeighborObs(
                neighbors=[
                    "same_front", "same_back",
                    "left_front", "left_back",
                    "right_front", "right_back",
                ],
                features=["distance", "rel_speed"],
                max_speed=max_speed,
                max_distance=max_distance,
                max_ttc=max_ttc,
            ),
        ])

        if reward == "fast":
            reward_function = sge.CompositeReward([
                sge.reward.StepPenalty(penalty=step_penalty),
                sge.reward.SpeedReward(max_speed=max_speed),
                sge.reward.TerminalReward(
                    w_crash=w_crash,
                    w_offroad=w_offroad,
                    w_arrived=w_arrived,
                    w_timeout=w_timeout,)
                ],
                weights=weights
            )

        if ego == "discrete":
            ego_controller = sge.ego.HighwayDiscreteEgo(
                    acc_value=acc_value,
                    dec_value=dec_value,
                    lc_duration=lc_duration)
        elif ego == "continuous":
            ego_controller = sge.ego.HighwayContinuousEgo(
                max_acc=max_acc,
                max_dec=max_dec,
                lc_duration=lc_duration,
                lane_threshold=lane_threshold,
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
                ])

        super().__init__(
            sumocfg_files=sumocfg_files,
            config=config,
            obs_builder=obs_builder,
            reward_function=reward_function,
            ego_controller=ego_controller,
            metrics_tracker=metrics_tracker,
        )
