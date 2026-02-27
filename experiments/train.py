import argparse

from src.infra.loaders.config_loader import load_config, load_config_from_model
from src.infra.builders.env_factory import build_env
from stable_baselines3.common.env_checker import check_env
from src.infra.builders.model_factory import build_model, load_model
from infra.trainer.trainer import train

DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_MODEL = None # "models/test_dqn_highway_2026-02-21_18-40-11/model.zip"
DEFAULT_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config YAML", default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="Path to pretrained model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", help="Random seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    if not args.config and not args.model:
        parser.error("You must provide either --config or --model")

    if args.model:
        cfg = load_config_from_model(args.model)
    else:
        cfg = load_config(args.config)

    env = build_env(cfg, seed=args.seed)

    print("\nCheck environment consistency...")
    check_env(env, warn=True)
    print("Done")

    if args.model:
        model = load_model(env, cfg, load_path=args.model, seed=args.seed)
    else:
        model = build_model(env, cfg, seed=args.seed)

    train(model, env, cfg)


if __name__ == "__main__":
    main()

