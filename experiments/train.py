import hydra
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb

import sumo_rl_ego as sre



@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):

    cfg_train = resolve_training_cfg(cfg)

    if not confirm_config(cfg_train):
        return
    
    # logger
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg_train, resolve=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
    )

    env = sre.make_vec_env(
        cfg_train.env,
        n_envs=cfg_train.resources.n_envs,
        base_seed=cfg_train.seed,
    )

    if cfg.checkpoint.load_path is None:

        model = sre.build_model(
            env,
            cfg_train.algo,
            seed=cfg_train.seed,
            device=cfg_train.resources.device
        )

    else:

        model = sre.load_model(
            env,
            cfg_train.algo,
            load_path=cfg.checkpoint.load_path,
            seed=cfg_train.seed,
            device=cfg_train.resources.device
        )

    model.tensorboard_log = "./tensorboard"

    model = sre.train(model, cfg.algo)

    model.save("model")

    print("\nTraining finished!")
    print("\nRun: tensorboard --logdir outputs")



def confirm_config(cfg):
    print(f"\n========== CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=" * 28 + "\n")

    answer = input("Continue? [y/N]: ").strip().lower()
    if answer not in ["y", "yes"]:
        print("Aborted.")
        return False

    return True


def resolve_training_cfg(cfg):
    """
    Restituisce il config effettivo da usare per il training.

    - se non c'è resume → usa cfg
    - se c'è resume → carica cfg_old dal run originale
      e applica SOLO gli override CLI
    """

    if cfg.checkpoint.load_path is None:
        return cfg

    cfg_old = sre.load_run(cfg.checkpoint.load_path)

    # copia per evitare side effects
    cfg_train = OmegaConf.create(OmegaConf.to_container(cfg_old, resolve=False))

    overrides = HydraConfig.get().overrides.task

    for o in overrides:
        if "=" not in o:
            continue

        key, value = o.split("=", 1)

        # skip parametri usati solo per il resume
        if key.startswith("checkpoint."):
            continue

        value_cfg = OmegaConf.create({"v": value})["v"]
        OmegaConf.update(cfg_train, key, value_cfg, merge=True)

    return cfg_train



if __name__ == "__main__":
    main()
