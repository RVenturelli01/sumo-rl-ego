from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from experiments.modules.my_ego import MyEgo
from experiments.modules.my_observation_v4 import MyObservation
from experiments.modules.my_reward import MyReward
from experiments.modules.my_kpi import MyKPI


config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    ego_id="ego",
    time_step=0.5,
    max_steps=50,
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

    # ---- Learning ----
    learning_rate=3e-4,      # più alto = converge prima
    gamma=0.98,              # leggermente più corto

    # ---- Replay ----
    buffer_size=20_000,     
    learning_starts=5_000,
    batch_size=64,

    # ---- Update ----
    train_freq=1,            
    gradient_steps=1,
    target_update_interval=5_000,

    # ---- Exploration ----
    exploration_fraction=0.4,
    exploration_final_eps=0.05,

    # ---- Network (KEY PART) ----
    policy_kwargs=dict(net_arch=[64, 64]),  

    verbose=1,
)


# ---- Train ----
try:
    model.learn(total_timesteps=20_000)
except KeyboardInterrupt:
    print("Training interrotto manualmente")
    model.save("dqn_sumo_agent_interrupt")

# ---- Save ----
model.save("dqn_sumo_agent")

env.close()

# tensorboard --logdir tensorboard_dqn

