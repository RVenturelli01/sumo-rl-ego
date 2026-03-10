import hydra
from omegaconf import DictConfig, OmegaConf
import traci

import sumo_rl_ego as sre


@hydra.main(version_base=None, config_path="configs", config_name="play.yaml")
def main(cfg: DictConfig):

    if cfg.model_path is not None:
        cfg_old = sre.load_run(cfg.model_path)
        env = sre.make_env(cfg_old.env, seed=cfg.seed, use_gui=True)
        model = sre.load_model(env=env, cfg=cfg_old.algo, seed=cfg.seed, load_path=cfg.model_path)
        policy = sre.policies.SB3Policy(model)

    elif cfg.policy_id is not None:
        env = sre.make_env(cfg.env, seed=cfg.seed, use_gui=True)
        policy = sre.load_policy(cfg.policy_id)

    else:
        raise ValueError("Either model_path or policy_id must be provided.")


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

        print("=" * 20 + "ACTION" + "=" * 20)
        env.ego_controller.print_action(action)
        print("=" * 20 + "OBSERVATION" + "=" * 20)
        env.obs_builder.print_obs(obs)
        print("=" * 50)

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