from sumo_rl_ego.configs.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from sumo_rl_ego.env.ego.discrete_long_lat_ego import DiscreteLongLatEgo
from sumo_rl_ego.env.observation.kin_lane_dist_observation import KinLaneDistObservation
from sumo_rl_ego.env.reward.higway_cruise_reward import HighwayCruiseReward



config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    use_gui=True,
    auto_start=False,
    ego_id="ego",
    ego_class=DiscreteLongLatEgo,
    obs_class=KinLaneDistObservation,
    reward_class=HighwayCruiseReward,
    #extra_sumo_args=["--quit-on-end", "true"],
)

env = SumoEnv(config)

obs, _ = env.reset()

for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print("Action:", action, "Obs:", obs, "Reward:", reward, "Info:", info)

    if terminated or truncated:
        break

env.close()
