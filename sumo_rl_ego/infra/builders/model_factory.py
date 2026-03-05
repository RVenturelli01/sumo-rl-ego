from stable_baselines3 import DQN, PPO

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
}

def build_model(env, cfg: dict, seed: int):
    print("\n[INFRA] Building a new model...")

    algo = ALGOS[cfg["algorithm"]]

    env.reset(seed=seed)

    model = algo(
        policy=cfg["policy"],
        env=env,
        **cfg["model"],
    )
    return model


def load_model(env, cfg: dict, load_path: str, seed: int):
    print(f"\n[INFRA] Loading rl model from {load_path}")

    algo = ALGOS[cfg["algorithm"]]

    env.reset(seed=seed)

    model = algo.load(load_path, env=env)  
    return model
