from pathlib import Path
import argparse
import time
import traci

from src.infra.loaders.config_loader import load_config
from src.infra.builders.env_factory import build_env
from src.infra.builders.model_factory import load_model
from src.infra.policy.sb3_policy import SB3Policy
from src.infra.loaders.class_loader import load_class


DEFAULT_MODEL = None # "outputs/models/test_dqn_highway_2026-02-21_22-43-05/model.zip"
DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_POLICY = "plugins.policies.my_policy.MyPolicy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=False, default=DEFAULT_MODEL)
    parser.add_argument("--policy", required=False, default=DEFAULT_POLICY)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config, args.model)

    cfg["sumo_config"]["use_gui"] = True
    env = build_env(cfg)

    if args.model:
        model = load_model(env, cfg, load_path=args.model)
        policy = SB3Policy(model=model)
    elif args.policy:
        policy = load_class(args.policy)()

    obs, _ = env.reset(seed=args.seed)

    # GUI tweaks (solo se SUMO GUI attiva)
    traci.gui.setSchema("View #0", "real world")
    traci.gui.trackVehicle("View #0", "ego")
    traci.gui.setZoom("View #0", 2000)

    print("Premi INVIO per fare uno step | Ctrl+C per uscire")

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

        input("\nPress Enter to step...\n")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(
                "Episode finished. ego status:",
                info.get("ego_status"),
                ", ep status:",
                info.get("ep_status"),
            )
            input("\nPress Enter to reset...\n")
            obs, _ = env.reset()
            traci.gui.trackVehicle("View #0", "ego")

    env.close()


if __name__ == "__main__":
    main()