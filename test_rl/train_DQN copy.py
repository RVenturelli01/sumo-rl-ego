from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from sumo_rl_ego.env.ego.highway_prof_ego import HighwayProfEgo
from sumo_rl_ego.env.observation.highway_prof_observation import HighwayProfObservation
from sumo_rl_ego.env.reward.higway_cruise_reward import HighwayCruiseReward


# ---- Create env ----
config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    ego_id="ego",
    ego_class=HighwayProfEgo,
    obs_class=HighwayProfObservation,
    reward_class=HighwayCruiseReward,
)

env = SumoEnv(config)

# Optional sanity check
check_env(env, warn=True)

# ---- Create agent ----
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=1_000,
    learning_starts=500,
    batch_size=128,
    gamma=0.99,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.2,
    verbose=1,
    tensorboard_log="./tensorboard_dqn/",
)

# ---- Train ----
model.learn(total_timesteps=5_000)

# ---- Save ----
model.save("dqn_sumo_agent")

env.close()

# tensorboard --logdir tensorboard_dqn