import argparse

from src.infra.loaders.config_loader import load_config
from src.infra.builders.env_factory import build_env
from stable_baselines3.common.env_checker import check_env
from src.infra.builders.model_factory import build_model, load_model
from infra.trainer.trainer import train

DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_MODEL = None # "models/test_dqn_highway_2026-02-21_18-40-11/model.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    print("\nLoading config file...")
    cfg = load_config(args.config, args.model)

    print("\nBuilding sumo gym environment...")
    env = build_env(cfg)

    print("\nCheck environment consistency...")
    check_env(env, warn=True)
    print("Done")

    if args.model:
        print(f"\nLoading rl model from {args.model}")
        model = load_model(env, cfg, load_path=args.model)
    else:
        print("\nBuilding a new model...")
        model = build_model(env, cfg)

    print("\nStart training phase...\n")
    train(model, env, cfg)


if __name__ == "__main__":
    main()

