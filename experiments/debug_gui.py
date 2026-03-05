from pathlib import Path
import argparse
import time
import traci

from sumo_rl_ego.infra.loaders.config_loader import load_config, load_config_from_model
from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.policy.sb3_policy import SB3Policy
from sumo_rl_ego.infra.loaders.class_loader import build_class


DEFAULT_MODEL = None # "outputs/models/test_dqn_highway_2026-02-21_22-43-05/model.zip"
DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_policy", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", help="Random seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    if not args.config_policy and not args.model:
        parser.error("You must provide either --config_policy or --model")

    if args.model:
        cfg = load_config_from_model(args.model)
    else:
        cfg = load_config(args.config_policy)

    # Build env (incapsula SumoConfig + builders vari)
    cfg["sumo_config"]["use_gui"] = True
    env = build_env(cfg, seed=args.seed)

    # Load trained model (SB3Policy wrapper gestito in model_factory)
    if args.model:
        model = load_model(env, cfg, load_path=args.model, seed=args.seed)
        policy = SB3Policy(model=model)
    elif args.config_policy:
        policy = build_class(cfg["policy"]["class"], cfg["policy"]["args"])

    obs, _ = env.reset(seed=args.seed)

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

        print("=" * 50)
        env.ego_controller.print_action(action)
        print("-" * 50)
        env.obs_builder.print_obs(obs)
        print("=" * 50)

        print("\nScenario: ",env.config.sumocfg_file)
        input("Press Enter to step...\n")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                "Episode finished. ego status:",
                info.get("ego_status"),
                ", ep status:",
                info.get("ep_status"),
            )
            print("Scenario: ",env.config.sumocfg_file)
            input("\nPress Enter to reset...\n")
            obs, _ = env.reset()
            traci.gui.trackVehicle("View #0", "ego")

    env.close()


if __name__ == "__main__":
    main()