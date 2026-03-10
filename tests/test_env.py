import numpy as np
import sumo_rl_ego as sre

env = sre.make_env("highway_discrete_v1", use_gui=False)

obs, info = env.reset()

print("\n--- ENV RESET ---")
print("Initial observation:", np.round(obs, 3))

total_reward = 0

for step in range(10):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"[step {step:02d}] "
        f"action={action} | "
        f"reward={reward:6.3f} | "
        f"term={terminated} | "
        f"trunc={truncated}"
    )

    if terminated or truncated:

        print("\n--- EPISODE ENDED ---")

        event = info.get("event", "unknown")
        metrics = info.get("metrics", {}).get("episode", {})

        print(f"event: {event}")

        if metrics:
            print("episode metrics:")
            for k, v in metrics.items():
                print(f"  {k:15s}: {v}")

        break


print("\n--- SUMMARY ---")
print(f"Total reward: {total_reward:.3f}")

env.close()

print("Test finished.")