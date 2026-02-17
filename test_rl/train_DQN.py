import traci
import numpy as np
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from sumo_rl_ego.env.ego.my_ego import MyEgo
from sumo_rl_ego.env.observation.my_observation import MyObservation
from sumo_rl_ego.env.reward.my_reward import MyReward
from sumo_rl_ego.env.kpi.my_kpi import MyKPI


config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    #use_gui=True,
    #auto_start=False,
    ego_id="ego",
    time_step=0.1
)

ego = MyEgo()
obs = MyObservation()
reward = MyReward()
kpi = MyKPI()
env = SumoEnv(config, ego, obs, reward, kpi)



from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Optional sanity check
check_env(env, warn=True)

# ---- Create agent ----
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=10_000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.98,
    train_freq=1,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=1,
    tensorboard_log="./tensorboard_dqn/",
)

# ---- Train ----
model.learn(total_timesteps=10_000)

# ---- Save ----
model.save("dqn_sumo_agent")

env.close()

# tensorboard --logdir tensorboard_dqn

