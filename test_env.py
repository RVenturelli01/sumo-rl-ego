from sumo_env.env import SumoEnv

env = SumoEnv()

obs, _ = env.reset()
print("Initial obs:", obs)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    print("Obs:", obs, "Reward:", reward)

env.close()
