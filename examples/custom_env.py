from pathlib import Path

import sumo_gym_ego as sge


scenario = Path("sumo_rl_ego/sumo_envs/scenarios/highway_fast_modified/highway.sumocfg")

env = sge.SumoEnv(
    sumocfg_files=[str(scenario)],
    config=sge.SumoConfig(use_gui=False, ego_id="ego", seed=0),
    ego_controller=sge.ego.HighwayDiscreteEgo(),
    obs_builder=sge.CompositeObservation([
        sge.obs.EgoSpeedObs(max_speed=50.0),
        sge.obs.EgoLaneObs(),
    ]),
    reward_function=sge.CompositeReward([
        sge.reward.StepPenalty(penalty=-0.2),
        sge.reward.SpeedReward(max_speed=50.0),
    ]),
    metrics_tracker=sge.CompositeMetricsTracker([
        sge.metrics.EgoFeatureMetrics(window=100),
    ]),
)

print(env)
env.close()
