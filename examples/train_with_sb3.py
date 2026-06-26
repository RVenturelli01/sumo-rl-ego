import sumo_gym_ego as sge
from stable_baselines3 import DQN


def main():

    env = sge.make_vec_env(
        "HighwayEgo-v0",
        n_envs=2,
        base_seed=0,
        reward="fast",
        ego="discrete",
        use_gui=False,
    )

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000)


if __name__ == "__main__":
    main()