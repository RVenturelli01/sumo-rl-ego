from sumo_rl_ego.configs.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from sumo_rl_ego.env.ego.discrete_long_ego import DiscreteLongEgo
from sumo_rl_ego.env.observation.kin_lane_dist_observation import KinLaneDistObservation
from sumo_rl_ego.env.reward.higway_cruise_reward import HighwayCruiseReward
from sumo_rl_ego.env.done.crash_done import CrashDone
from sumo_rl_ego.env.done.offroad_done import OffRoadDone
from sumo_rl_ego.env.done.horizon_done import HorizonDone
from sumo_rl_ego.env.done.combined_done import CombinedDone



config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    use_gui=True,
    #auto_start=False,
    ego_id="ego",
    ego_class=DiscreteLongEgo,
    obs_class=KinLaneDistObservation,
    reward_class=HighwayCruiseReward,
    done_class=lambda: CombinedDone(HorizonDone()),
)

env = SumoEnv(config)

obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, _ = env.step(action)

    print("Obs:", obs, "Reward:", reward)

    if done:
        break

env.close()
