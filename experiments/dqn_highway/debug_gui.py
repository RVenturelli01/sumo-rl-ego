from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import traci
import numpy as np
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from experiments.modules.my_ego import MyEgo
from experiments.modules.my_observation_v2 import MyObservation
from experiments.modules.my_reward import MyReward
from experiments.modules.my_kpi import MyKPI


config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    use_gui=True,
    ego_id="ego",
    time_step=0.5,
)


env = SumoEnv(config, 
              ego = MyEgo(), 
              obs_builder = MyObservation(), 
              reward_fn = MyReward())


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

    if terminated or truncated:
        print("Episode finished:", info)
        config.seed += 1  # change seed for next episode
        obs, _ = env.reset()
        traci.gui.trackVehicle("View #0", "ego")
        input("Premi invio per chiudere...") # per debuggare, da rimuovere


    print("Action: ", action, "Obs: ", obs, "  Speed:", traci.vehicle.getSpeed("ego"))
    

env.close()
