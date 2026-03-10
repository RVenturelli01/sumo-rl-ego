from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3

ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def build_model(env, cfg: DictConfig, seed: int, device: str):
    print("[SRE] Building model...")

    algo_name = cfg.algorithm
    algo_cls = ALGO_REGISTRY[algo_name]

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    env.reset()

    model = algo_cls(
        env=env,
        seed=seed,
        device=device,
        **model_kwargs,
    )

    return model


def load_model(env, cfg: DictConfig, load_path: str, seed: int, device: str = "cpu"):
    print("[SRE] Loading model from path:", load_path)

    algo_name = cfg.algorithm
    algo_cls = ALGO_REGISTRY[algo_name]

    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)

    env.reset()

    model = algo_cls.load(
        load_path,
        env=env,
        device=device,
        **model_kwargs,
    )

    return model