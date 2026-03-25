import sumo_rl_ego as sre

print("list of the available environments: ", sre.list_envs())

env = sre.make_env(
    "HighwayEgo-v0",
    seed=0,
    reward="fast",
    ego="discrete",
    use_gui=False,
)

print(env)
env.close()
