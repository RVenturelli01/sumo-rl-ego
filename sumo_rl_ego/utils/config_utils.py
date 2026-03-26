from pathlib import Path

import hydra

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



def save_outputs(cfg, model) -> None:
    # === Resolve output directory (Hydra run dir) ===
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # === CONFIG ===
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path, resolve=True)
    print(f"Saved config to {config_path}")

    # === MODEL ===
    model_path = output_dir / "model.zip"
    model.save(model_path)
    print(f"Saved model to {model_path}")

    # === WANDB ===
    if cfg.wandb.enabled and wandb.run is not None:
        wandb.save(str(model_path))

   # === REPLAY BUFFER ===
    if hasattr(model, "save_replay_buffer"):
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        path = output_dir / "replay_buffer.pkl"

        model.save_replay_buffer(path)
        print(f"Saved replay buffer to {path}")
    else:
        print("Replay buffer not supported")

        

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
            device="cpu",
        )
        return sre.ModelPolicy(model)
    
    if cfg.source.model_id is not None:
        return sre.load_policy(cfg.source.model_id, env=env)
    
    if cfg.source.policy_id is not None:
        return sre.load_policy(cfg.source.policy_id)
    
    if cfg.source.policy_class is not None:
        return instantiate(cfg.source.policy_class)
    

