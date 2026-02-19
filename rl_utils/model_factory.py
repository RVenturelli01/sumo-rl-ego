from stable_baselines3 import DQN, PPO

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
}

def build_model(env, cfg: dict):
    algo = ALGOS[cfg["algorithm"]]

    model = algo(
        policy=cfg["policy"],
        env=env,
        **cfg["model"],
    )
    return model
