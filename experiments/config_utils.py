from pathlib import Path

import numpy as np
import sumo_rl_ego as sre
import wandb

from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3


ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def resolve_paths(cfg):
    if cfg.source is not None and cfg.source.model_path is not None:
        cfg.source.model_path = str(Path(get_original_cwd()) / cfg.source.model_path)


def print_eval_cfg(cfg, title):
    print(f"\n========== {title} CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id}")
    print(f"Environment arguments: {cfg.env.kwargs}")

    def is_active(k):
        v = cfg.source.get(k)
        return v is not None and (k != "policy_class" or v.get("__target__") is not None)

    fields = ["model_id", "model_path", "policy_id", "policy_class"]
    active = [k for k in fields if is_active(k)]

    print(f"{active[0]}: {cfg.source[active[0]]}") 
    print(f"Number of episodes: {cfg.n_episodes}")
    print("\n=============================================\n")

    

def print_play_cfg(cfg, title):
    print(f"\n========== {title} CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id}")
    print(f"Environment arguments: {cfg.env.kwargs}")

    def is_active(k):
        v = cfg.source.get(k)
        return v is not None and (k != "policy_class" or v.get("__target__") is not None)

    fields = ["model_id", "model_path", "policy_id", "policy_class"]
    active = [k for k in fields if is_active(k)]

    print(f"{active[0]}: {cfg.source[active[0]]}") 
    print(f"manual: {cfg.manual}")
    print("\n=============================================\n")

    
def confirm_cfg():
    while True:
        user_input = input("Proceed with this configuration? (y/n): ").strip().lower()
        if user_input == 'y':
            break
        elif user_input == 'n':
            print("Aborting run.")
            exit(0)
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def init_wandb(cfg):
    if not cfg.wandb.enabled:
        return None

    return wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        **OmegaConf.to_container(cfg.wandb.kwargs, resolve=True),
    )


def load_cfg_from_model_path(model_path):
    model_path = Path(model_path)
    model_dir = model_path.parent

    hydra_cfg = model_dir / ".hydra" / "config.yaml"
    plain_cfg = model_dir / "config.yaml"

    if hydra_cfg.exists():
        return OmegaConf.load(hydra_cfg)
    if plain_cfg.exists():
        return OmegaConf.load(plain_cfg)

    raise FileNotFoundError(
        f"No config found in {model_dir} (expected .hydra/config.yaml or config.yaml)"
    )



def save_outputs(cfg: DictConfig, model) -> None:
    # === CONFIG ===
    config_path = Path(cfg.save.config_dir) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Saved config to {config_path}")

    if wandb.run is not None:
        wandb.save(str(model_path))

    # === MODEL ===
    model_path = Path(cfg.save.model_dir) / "model.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # === REPLAY BUFFER ===
    if hasattr(model, "save_replay_buffer") and cfg.save.replay_buffer_dir is not None:
        replay_buffer_path = Path(cfg.save.replay_buffer_dir) / "replay_buffer.pkl"
        replay_buffer_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_replay_buffer(replay_buffer_path)
        print(f"Saved replay buffer to {replay_buffer_path}")
    else:
        print("Warning: replay buffer saving is not supported for this algorithm")

        

def check_source_cfg(cfg: DictConfig) -> None:
    sources = [
    cfg.source.get("model_path"),
    cfg.source.get("model_id"),
    cfg.source.get("policy_id"),
    cfg.source.get("policy_class", {}).get("__target__"),
    ]
    count = sum(x is not None for x in sources)
    if count == 0:
        raise ValueError("At least one source (model_path, model_id, policy_id, policy_class) must be specified in the config.")
    elif count > 1:
        raise ValueError("Only one source (model_path, model_id, policy_id, policy_class) can be specified in the config.")
    

def load_policy_from_cfg(cfg: DictConfig, env=None) -> DictConfig:
    if cfg.source.model_path is not None:
        source_cfg = load_cfg_from_model_path(cfg.source.model_path)
        algo_cls = ALGO_REGISTRY[source_cfg.model.algo]
        model = algo_cls.load(
            path=cfg.source.model_path,
            env=env,
        )
        return sre.policy_from_model(model)
    
    if cfg.source.model_id is not None:
        return sre.load_model(cfg.source.model_id, env=env)
    
    if cfg.source.policy_id is not None:
        return sre.load_policy(cfg.source.policy_id)
    
    if cfg.source.policy_class is not None:
        return instantiate(cfg.source.policy_class)
    

