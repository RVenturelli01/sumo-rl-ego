from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN, PPO

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
}

def build_model(env, cfg: DictConfig, seed: int):
    print("\n[INFRA] Building a new model...")

    algo = ALGOS[cfg.algorithm]
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    env.reset()

    model = algo(
        env=env,
        seed=seed,
        **model_kwargs,
    )
    return model


def load_model(env, cfg: DictConfig, load_path: str, seed: int):
    print(f"\n[INFRA] Loading rl model from {load_path}")

    algo = ALGOS[cfg.algorithm]
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    env.reset()

    model = algo.load(
        load_path,
        env=env,
        seed=seed,
        custom_objects=model_kwargs,
    )

    return model
