from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

import traci
import numpy as np
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from experiments.modules.my_ego import MyEgo
from experiments.modules.my_observation import MyObservation
from experiments.modules.my_reward import MyReward
from experiments.modules.my_metrics import MyKPI


config = SumoConfig(
    sumocfg_file="networks/highway_fast/highway.sumocfg",
    use_gui=True,
    ego_id="ego",
    time_step=0.5,
)

ego = MyEgo()
obs_builder = MyObservation()
reward_fn = MyReward()
env = SumoEnv(config, ego=ego, obs_builder=obs_builder, reward_fn=reward_fn)


# ---- Load trained agent ----
model = DQN.load("dqn_sumo_agent")

obs, _ = env.reset()

print(traci.vehicle.getIDList())

# opzionale: fai seguire l'ego
traci.gui.trackVehicle("View #0", "ego")
traci.gui.setZoom("View #0", 1800)
input("Premi invio per chiudere...") # per debuggare, da rimuovere

# deterministic=True = niente epsilon-greedy
action, _ = model.predict(obs, deterministic=True)

# ---- Rollout ----
while True:

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished:", info)
        obs, _ = env.reset()
        traci.gui.trackVehicle("View #0", "ego")
        input("Premi invio per chiudere...") # per debuggare, da rimuovere

    # deterministic=True = niente epsilon-greedy
    action, _ = model.predict(obs, deterministic=True)

    print("="*50)
    print(f"Action: {ego.print_action(action)}")
    print(f"Ego real speed: {traci.vehicle.getSpeed('ego'):.2f} m/s")
    print(obs_builder.print_obs(obs))

    
env.close()
