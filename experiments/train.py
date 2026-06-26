import hydra
import wandb
import sumo_gym_ego as sge
import sumo_rl_ego as sre

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

from sumo_rl_ego.utils import (
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
    
    
def print_train_cfg(cfg):
    print(f"\n========== TRAIN CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id} (x{cfg.env.n_envs} envs)")
    print(f"Environment arguments: {cfg.env.kwargs}")
    print(f"Algorithm: {cfg.model.algo}")
    print(f"Timesteps: {cfg.learn.kwargs.total_timesteps}")
    print("\n=============================================\n")
    
    

@hydra.main(version_base=None, config_path="configs/train", config_name="dqn.yaml")
def main(cfg: DictConfig) -> None:
    _ = HydraConfig.get().runtime.output_dir

    print_train_cfg(cfg)

    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        **OmegaConf.to_container(cfg.wandb.kwargs, resolve=True),
    ) if cfg.wandb.enabled else None
    env = None

    try:
        print("Creating environment...")
        env = sge.make_vec_env(
            cfg.env.id,
            n_envs=cfg.env.n_envs,
            base_seed=cfg.run.seed,
            **cfg.env.kwargs
        )

        print("Initializing model...")
        model_kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
        algo_cls = ALGO_REGISTRY[cfg.model.algo]
        model = algo_cls(
            env=env, 
            **model_kwargs
        )

        print("Starting training...\n")
        model.learn(
            callback=CustomLoggingCallback(),
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
