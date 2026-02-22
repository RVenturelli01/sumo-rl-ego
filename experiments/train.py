import argparse

from src.utils.config_loader import load_config
from src.core.env_factory import build_env
from stable_baselines3.common.env_checker import check_env
from src.core.model_factory import build_model, load_model
from src.core.trainer import train

DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_MODEL = None # "models/test_dqn_highway_2026-02-21_18-40-11/model.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    cfg = load_config(args.config, args.model)
    env = build_env(cfg)

    check_env(env, warn=True)

    if args.model:
        model = load_model(env, cfg, load_path=args.model)
    else:
        model = build_model(env, cfg)

    train(model, env, cfg)


if __name__ == "__main__":
    main()

