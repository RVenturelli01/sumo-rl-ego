import hydra
from omegaconf import DictConfig, OmegaConf
import traci

from stable_baselines3.common.env_checker import check_env

from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.policy.sb3_policy import SB3Policy


@hydra.main(version_base=None, config_path="configs", config_name="play")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Force GUI mode
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    env_cfg["sumo_config"]["use_gui"] = True
    env_cfg = OmegaConf.create(env_cfg)

    env = build_env(env_cfg, seed=cfg.seed)

    print("\nCheck environment consistency...")
    check_env(env, warn=True)
    print("Done")

    if cfg.model_path:
        model = load_model(env, cfg.rl, load_path=cfg.model_path, seed=cfg.seed)
        policy = SB3Policy(model=model)
    elif cfg.policy._target_:
        policy = hydra.utils.instantiate(cfg.policy)
    else:
        raise ValueError("You must provide either model_path or policy._target_")

    obs, _ = env.reset(seed=cfg.seed)

    # GUI tweaks (solo se SUMO GUI attiva)
    traci.gui.setSchema("View #0", "real world")
    traci.gui.trackVehicle("View #0", "ego")
    traci.gui.setZoom("View #0", 1000)

    print("\nScenario: ",env.config.sumocfg_file)
    print("Premi INVIO per fare uno step | Ctrl+C per uscire\n")

    # ======================
    # Manual rollout loop
    # ======================
    while True:
        action = policy.predict(obs)

        # print("=" * 20 + "ACTION" + "=" * 20)
        # env.ego_controller.print_action(action)
        # print("=" * 20 + "OBSERVATION" + "=" * 20)
        # env.obs_builder.print_obs(obs)
        # print("=" * 50)

        # print("\nScenario: ",env.config.sumocfg_file)
        input("Press Enter to step...\n")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                "Episode finished. Simulation status:",
                info.get("status"),
                ", event:",
                info.get("event"),
            )
            print("Scenario: ",env.config.sumocfg_file)
            input("\nPress Enter to reset...\n")
            obs, _ = env.reset()
            traci.gui.trackVehicle("View #0", "ego")

    env.close()


if __name__ == "__main__":
    main()