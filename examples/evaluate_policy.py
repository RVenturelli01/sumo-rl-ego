import sumo_gym_ego as sge
import sumo_rl_ego as sre

print("list of the available environments: ", sge.sumo_envs.registry.list_envs())
print("list of the available policies: ", sre.list_policies())
print("list of the available models: ", sre.list_models())

env = sge.make_env(
    "HighwayEgo-v0",
    seed=0,
    reward="fast",
    ego="discrete",
    use_gui=False,
)
policy = sre.load_policy('dqn_v0')

info = sre.run_episode(env, policy, seed=0)
print(info)
env.close()
