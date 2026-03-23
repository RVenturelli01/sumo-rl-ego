import traci
import time

def play_policy(env, policy, manual=False, max_episodes=None):
    
    episode_idx = 0

    while max_episodes is None or episode_idx < max_episodes: 

        obs, _ = env.reset()
        policy.reset()  

        traci.gui.setSchema("View #0", "real world")
        traci.gui.trackVehicle("View #0", "ego")
        traci.gui.setZoom("View #0", 1700)     

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.predict(obs)

            if manual:
                print("=" * 20 + "ACTION" + "=" * 20)
                if hasattr(env, "ego_controller") and hasattr(env.ego_controller, "print_action"):
                    env.ego_controller.print_action(action)
                else:
                    print(action)

                print("=" * 20 + "OBSERVATION" + "=" * 20)
                if hasattr(env, "obs_builder") and hasattr(env.obs_builder, "print_obs"):
                    env.obs_builder.print_obs(obs)
                else:
                    print(obs)

                print("=" * 50)
                input("Press Enter to step...\n")
            else:
                time.sleep(0.1) 

            obs, reward, terminated, truncated, info = env.step(action)

        print("Episode finished:", info.get("event"))

        if manual and (max_episodes is None or episode_idx + 1 < max_episodes):
            input("Press Enter to reset...\n")

        episode_idx += 1

