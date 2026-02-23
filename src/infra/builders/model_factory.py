from stable_baselines3 import DQN, PPO

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
}

def build_model(env, cfg: dict):
    algo = ALGOS[cfg["algorithm"]]

    env.reset(seed=cfg["env"]["seed"])

    model = algo(
        policy=cfg["policy"],
        env=env,
        **cfg["model"],
    )
    return model


def load_model(env, cfg: dict, load_path: str):
    algo = ALGOS[cfg["algorithm"]]

    env.reset(seed=cfg["env"]["seed"])

    model = algo.load(load_path, env=env)  
    return model
