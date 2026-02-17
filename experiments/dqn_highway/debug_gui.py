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
    use_gui=True,
    #auto_start=False,
    ego_id="ego",
    #seed=0,
    time_step=0.1,
    # extra_sumo_args=[
    #     "--delay", "100"
    # ],
)

ego = MyEgo()
obs = MyObservation()
reward = MyReward()
kpi = MyKPI()
env = SumoEnv(config, ego, obs, reward, kpi)


from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# ---- Load trained agent ----
model = DQN.load("dqn_sumo_agent")

obs, _ = env.reset()

# opzionale: fai seguire l'ego
traci.gui.trackVehicle("View #0", "ego")
traci.gui.setZoom("View #0", 1500)
input("Premi invio per chiudere...") # per debuggare, da rimuovere

# ---- Rollout ----
while True:
    # deterministic=True = niente epsilon-greedy
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    print("Action: ", action, "  Speed:", traci.vehicle.getSpeed("ego"), "  Reward: ", reward)
    print("Obs: ", obs)

    if terminated or truncated:
        print("Episode finished:", info)
        config.seed += 1  # change seed for next episode
        obs, _ = env.reset()
        traci.gui.trackVehicle("View #0", "ego")
        input("Premi invio per chiudere...") # per debuggare, da rimuovere

env.close()
