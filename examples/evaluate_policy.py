import sumo_rl_ego as sre

print("list of the available policies: ", sre.list_policies())
print("list of the available models: ", sre.list_models())

env = sre.make_env(
    "HighwayEgo-v0",
    seed=0,
    reward="fast",
    ego="discrete",
    use_gui=False,
)
policy = sre.load_policy('FastPolicy-v0')

info = sre.run_episode(env, policy, seed=0)
print(info)
env.close()
