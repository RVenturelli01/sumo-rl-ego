import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
from wandb.integration.sb3 import WandbCallback
import wandb

import sumo_rl_ego as sre


ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def confirm_start():
    answer = input("\nStart training? [y/N]: ").strip().lower()
    if answer not in ["y", "yes"]:
        print("Training aborted by user.")
        exit()


def apply_overrides(model, overrides):
    for key, value in overrides.items():
        if value is None:
            continue

        if hasattr(model, key):
            setattr(model, key, value)
            print(f"{key} -> {value}")

            if key == "learning_rate":
                model._setup_lr_schedule()
        else:
            print(f"Warning: '{key}' not in model")


def train(cfg):
    print("\n========== TRAINING CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=====================================\n")
    confirm_start()


    print("Initializing Weights & Biases...")
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.wandb_kwargs)


    print("Creating environment...")
    env = sre.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=cfg.seed,
        **cfg.env.env_args,
    )

    print("Initializing algorithm...")
    algo_cls = ALGO_REGISTRY[cfg.algo.type]
    model = algo_cls(env=env, **cfg.algo.algo_kwargs)

    print("Starting training...\n")
    model.learn(
        callback=sre.CustomLogsCallback(),
        **cfg.learn_kwargs,
    )

    print("Training finished.")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = run_dir / "model.zip"
    model.save(model_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    run.finish()

    print("Run completed successfully.\n")



def finetune(cfg):
    old_cfg = OmegaConf.load(to_absolute_path(cfg.cfg_path))
    cfg = OmegaConf.merge(old_cfg, cfg)
    cfg.experiment.model_dir = to_absolute_path(cfg.model_dir)
    cfg.experiment.learn_kwargs.reset_num_timesteps = False


    print("\n========== FINE-TUNING CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=====================================\n")
    confirm_start()

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    print("Initializing Weights & Biases...")
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
        **cfg.wandb_kwargs,
    )    

    print("Creating environment...")
    env = sre.make_vec_env(
        cfg.env.id,
        n_envs=cfg.env.n_envs,
        base_seed=cfg.seed,
    )

    print("Initializing algorithm...")
    algo_cls = ALGO_REGISTRY[cfg.algo.type]
    model = algo_cls.load(path=cfg.experiment.model_path, env=env)
    apply_overrides(model, cfg.override)

    print("Starting training...\n")
    model.learn(
        callback=sre.CustomLogsCallback(),
        **cfg.learn_kwargs,
    )

    print("Training finished.")

    model_path = run_dir / "model.zip"
    model.save(model_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    run.finish()

    print("Run completed successfully.")




@hydra.main(version_base=None, config_path="configs/train", config_name="dqn.yaml")
def main(cfg: DictConfig):
    
    if cfg.finetuning:
        finetune(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()