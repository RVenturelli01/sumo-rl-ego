import traci
import numpy as np
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv



config = SumoConfig(
    sumocfg_file="scenarios/highway_fast/highway.sumocfg",
    use_gui=True,
    #auto_start=False,
    ego_id="ego",
    #seed=42,
    time_step=0.1,
    # extra_sumo_args=[
    #     "--delay", "100"
    # ],
)

env = SumoEnv(config)

obs, _ = env.reset()
traci.gui.trackVehicle("View #0", "ego")
traci.gui.setZoom("View #0", 2000)
input("Premi invio per chiudere...") # per debuggare, da rimuovere


for _ in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended. Resetting environment.")
        print("\nInfo:", info)
        env.config.seed += 1  # Change seed for next episode
        obs = env.reset()
        input("Premi invio per chiudere...") # per debuggare, da rimuovere

    traci.gui.trackVehicle("View #0", "ego")


print("test ended.")
env.close()
