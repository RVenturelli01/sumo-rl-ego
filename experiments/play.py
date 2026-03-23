import hydra
import sumo_rl_ego as sre

from omegaconf import DictConfig

from .config_utils import (
    print_play_cfg,
    confirm_cfg,
    check_source_cfg,
    load_policy_from_cfg,
)



@hydra.main(version_base=None, config_path="configs", config_name="play.yaml")
def main(cfg: DictConfig):
    check_source_cfg(cfg)

    print_play_cfg(cfg, "PLAY")
    confirm_cfg()

    print("Loading environment...")
    env = sre.make_env(
        cfg.env.id, 
        seed=cfg.seed, 
        **cfg.env.kwargs
    )

    print("Loading policy...")
    policy = load_policy_from_cfg(cfg)

    print("Starting play...")
    sre.play_policy(
        env,
        policy,
        seed=cfg.seed,
        manual=cfg.manual,
    )

    env.close()


if __name__ == "__main__":
    main()
