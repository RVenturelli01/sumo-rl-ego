import numpy as np
from sumo_rl_ego.sumo_gym_ego.core.config import SumoConfig
from sumo_rl_ego.sumo_gym_ego.env import SumoEnv


config = SumoConfig(
    sumocfg_file="scenarios/highway_fast/highway.sumocfg",
    use_gui=False,
    ego_id="ego",
    time_step=0.1,
)

env = SumoEnv(sumocfg_files=[config.sumocfg_file], 
              config=config)

obs, info = env.reset()
print("Initial observation:", obs)

total_reward = 0

for step in range(10):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    print(
        f"step={step} | "
        f"action={action} | "
        f"reward={reward:.3f} | "
        f"terminated={terminated} | "
        f"truncated={truncated}"
    )

    if terminated or truncated:
        print("Episode ended")
        print("Final info:", info)
        break


print("Total reward:", total_reward)

env.close()
print("Test finished.")