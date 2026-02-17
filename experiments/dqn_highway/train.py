from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from components.my_ego import MyEgo
from components.my_observation import MyObservation
from components.my_reward import MyReward
from components.my_kpi import MyKPI


config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    ego_id="ego",
    time_step=0.5,
    max_steps=100,
)

env = SumoEnv(config, 
              ego = MyEgo(), 
              obs_builder = MyObservation(), 
              reward_fn = MyReward())


# Optional sanity check
check_env(env, warn=True)

# ---- Create agent ----
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=2e-4,
    buffer_size=10_000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.95,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=2000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[32, 32]),
    verbose=1,
    tensorboard_log=None,
)


# ---- Train ----
model.learn(total_timesteps=20_000)

# ---- Save ----
model.save("dqn_sumo_agent")

env.close()

# tensorboard --logdir tensorboard_dqn

