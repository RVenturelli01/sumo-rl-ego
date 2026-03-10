import traci
import sumo_rl_ego as sre


env = sre.make_env("highway_discrete_v1", use_gui=True)

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
