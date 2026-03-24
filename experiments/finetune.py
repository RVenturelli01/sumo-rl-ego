import hydra
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.utils import LinearSchedule


from sumo_rl_ego.utils import (
    init_wandb, 
    confirm_cfg,
    save_outputs,
    CustomLoggingCallback,
)


ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}


def print_finetune_cfg(cfg):
    print(f"\n========== FINE-TUNE CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id} (x{cfg.env.n_envs} envs)")
    print(f"Environment arguments: {cfg.env.kwargs}")
    print(f"Algorithm: {cfg.model.algo}")
    print(f"Model overrides: {cfg.model.overrides}")
    print(f"Timesteps: {cfg.learn.kwargs.total_timesteps}")
    print("\n=============================================\n")
    
def apply_model_overrides(model, overrides_cfg: DictConfig) -> None:
    overrides = OmegaConf.to_container(overrides_cfg, resolve=True)

    for key, value in overrides.items():
        if value is None:
            continue

        if not hasattr(model, key):
            print(f"Warning: model has no attribute '{key}', skipping override")
            continue

        if key == "learning_rate":
            model.learning_rate = value
            model._setup_lr_schedule()

            # Update optimizer learning rate (important!)
            for param_group in model.policy.optimizer.param_groups:
                param_group["lr"] = model.lr_schedule(1.0)

            print(f"override: learning_rate -> {value}")

        else:
            setattr(model, key, value)
            print(f"override: {key} -> {value}")



@hydra.main(version_base=None, config_path="configs/finetune", config_name="dqn_ft.yaml")
def main(cfg: DictConfig) -> None:
    _ = HydraConfig.get().runtime.output_dir

    print_finetune_cfg(cfg)
    confirm_cfg()
    
    run = init_wandb(cfg)
    env = None

    try:
        print("Creating environment...")
        env = sre.make_vec_env(
            cfg.env.id, 
            n_envs=cfg.env.n_envs, 
            base_seed=cfg.run.seed, 
            **cfg.env.kwargs
        )

        print("Initializing model...")
        algo_cls = ALGO_REGISTRY[cfg.model.algo]
        model = algo_cls.load(
            path=cfg.source.model_path,
            env=env,
            custom_objects=OmegaConf.to_container(cfg.model.overrides, resolve=True),
        )

        if cfg.source.replay_buffer_path is not None:
            model.load_replay_buffer(cfg.source.replay_buffer_path)

        # apply_model_overrides(model, cfg.model.overrides)
        
        print("exploration_schedule updated with overrides:", model.exploration_schedule.__dict__)
        
        print("Starting training...\n")
        model.learn(
            callback=CustomLoggingCallback(window_size=cfg.learn.rolling_window),
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
