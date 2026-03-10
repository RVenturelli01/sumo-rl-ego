import hydra
from pathlib import Path
from omegaconf import DictConfig

from sumo_rl_ego.sumo_gym_ego.env import SumoEnv
from sumo_rl_ego.sumo_gym_ego.core.config import SumoConfig
from .utils.project import find_repo_root
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.vec_env import SubprocVecEnv


def _resolve_sumocfg_paths(paths: list[str]) -> list[str]:
    """Resolve sumocfg paths relative to the repo root."""
    root = find_repo_root()
    resolved = []
    for p in paths:
        full = root / p
        if not full.exists():
            raise FileNotFoundError(f"Scenario config not found: {full}")
        resolved.append(str(full))
    return resolved


def build_env(cfg: DictConfig, seed: int) -> SumoEnv:
    print("\n[INFRA] Building sumo gym environment...")

    sumo_cfg = SumoConfig(**cfg.sumo_config, seed=seed)

    ego_class = hydra.utils.instantiate(cfg.ego)
    obs_class = hydra.utils.instantiate(cfg.obs)
    reward_class = hydra.utils.instantiate(cfg.reward)
    metrics_class = hydra.utils.instantiate(cfg.metrics)

    env = SumoEnv(
        sumocfg_files=_resolve_sumocfg_paths(cfg.sumocfg_files),
        config=sumo_cfg,
        ego_controller=ego_class,
        obs_builder=obs_class,
        reward_function=reward_class,
        metrics_tracker=metrics_class,
    )

    print("Env loaded with the following components:")
    print(f"  ego:     {cfg.ego._target_}")
    print(f"  obs:     {cfg.obs._target_}")
    print(f"  reward:  {cfg.reward._target_}")
    print(f"  metrics: {cfg.metrics._target_}")

    print("\nCheck environment consistency...")
    check_env(env, warn=True)
    print("Done")

    return env



def build_vec_env(cfg: DictConfig, n_envs: int, base_seed: int = 0):

    print(f"\n[INFRA] Building vectorized env ({n_envs} envs)...")

    def make_env(rank: int):
        def _init():
            seed = base_seed + rank
            env = build_env(cfg, seed)
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    print(f"Vectorized environment ready ({n_envs} processes).")

    return env
