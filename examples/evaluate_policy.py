import sumo_rl_ego as sre


env = sre.make_env(
    "HighwayEgo-v0",
    seed=0,
    reward="fast",
    ego="discrete",
    use_gui=False,
)
policy = sre.load_policy("safe_speed_v1")

result = sre.evaluate_policy(env, policy, n_episodes=3, seed=0)
print(result)
env.close()
