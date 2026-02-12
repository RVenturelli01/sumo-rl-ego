from sumo_rl_ego.configs.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from sumo_rl_ego.env.ego.longitudinal_ego import LongitudinalEgo
from sumo_rl_ego.env.observation.simple_observation import SimpleObservation
from sumo_rl_ego.env.reward.speed_reward import SpeedReward
from sumo_rl_ego.env.done.collision_done import CollisionDone



config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    use_gui=True,
    auto_start=False,
    ego_id="ego",
    ego_class=LongitudinalEgo,
    obs_class=SimpleObservation,
    reward_class=SpeedReward,
    done_class=CollisionDone,
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
