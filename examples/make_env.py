import sumo_gym_ego as sge
import sumo_rl_ego as sre

print("list of the available environments: ", sge.sumo_envs.registry.list_envs())

env = sge.make_env(
    "HighwayEgo-v0",
    seed=0,
    reward="fast",
    ego="discrete",
    use_gui=False,
)

print(env)
env.close()
