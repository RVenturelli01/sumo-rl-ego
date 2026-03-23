import hydra
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3


from sumo_rl_ego.utils import (
    init_wandb, 
    confirm_cfg,
    load_cfg_from_model_path,
    save_outputs,
    WandbCustomCallback,
)


ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}



def build_finetune_cfg(cfg: DictConfig) -> DictConfig:
    source_cfg = load_cfg_from_model_path(cfg.source.model_path)
    final_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    final_cfg.env = source_cfg.env
    final_cfg.model.algo = source_cfg.model.algo
    return final_cfg


def apply_model_overrides(model, overrides_cfg: DictConfig) -> None:
    overrides = OmegaConf.to_container(overrides_cfg, resolve=True)

    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(model, key):
            print(f"Warning: model has no attribute '{key}', skipping override")
            continue

        setattr(model, key, value)
        print(f"override: {key} -> {value}")

        if key == "learning_rate":
            model._setup_lr_schedule()


@hydra.main(version_base=None, config_path="configs/finetune", config_name="dqn.yaml")
def main(cfg: DictConfig) -> None:
    _ = HydraConfig.get().runtime.output_dir
    cfg = build_finetune_cfg(cfg)

    print_config(cfg, "FINE-TUNE")
    confirm_config()
    
    run = init_wandb(cfg)
    env = None

    try:
        print("Creating environment...")
        env = sre.make_vec_env(
            cfg.env.id, 
            n_envs=cfg.env.n_envs, 
            base_seed=cfg.seed, 
            **cfg.env.kwargs
        )

        print("Initializing model...")
        algo_cls = ALGO_REGISTRY[cfg.model.algo]
        model = algo_cls.load(
            load_path=cfg.source.model_path,
            env=env,
        )
        apply_model_overrides(model, cfg.model.overrides)

        print("Starting training...\n")
        model.learn(
            callback=WandbCustomCallback(window_size=1000),
            **cfg.learn.kwargs,
        )

        print("\nTraining finished.")
        save_outputs(cfg, model)
        print("Run completed successfully.\n")

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
