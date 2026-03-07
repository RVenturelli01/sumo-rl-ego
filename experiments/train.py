import sys
import os

import argparse
from stable_baselines3.common.monitor import Monitor

from sumo_rl_ego.infra.loaders.config_loader import load_config
from sumo_rl_ego.infra.builders.env_factory import build_env
from stable_baselines3.common.env_checker import check_env
from sumo_rl_ego.infra.builders.model_factory import build_model, load_model
from sumo_rl_ego.infra.trainer.trainer import train
from sumo_rl_ego.infra.utils.artifact_manager import save_model, save_config
from sumo_rl_ego.infra.utils.run_manager import create_run



DEFAULT_CONFIG = None
DEFAULT_SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config YAML", default=DEFAULT_CONFIG)
    parser.add_argument("--seed", help="Random seed", default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    if not args.config:
        parser.error("You must provide --config")

    config_exp = load_config(args.config)
    mode = config_exp["mode"]


    if mode == "train":
        config_env = load_config(config_exp["env"])
        config_rl = load_config(config_exp["rl"])

        env = build_env(config_env, seed=args.seed)

        print("\nCheck environment consistency...")
        check_env(env, warn=True)
        print("Done")
        
        model = build_model(env, config_rl, seed=args.seed)

        run_dir, tb_dir = create_run(config_exp)

        env = Monitor(env, str(run_dir))
        model.tensorboard_log = str(tb_dir)

        model = train(model, config_rl)

        save_model(model, run_dir)
        save_config(config_exp, run_dir, "config_exp.yaml")
        save_config(config_env, run_dir, "config_env.yaml")
        save_config(config_rl, run_dir, "config_rl.yaml")


    elif mode == "finetuning":
        model_dir = os.path.dirname(config_exp["model"])
        config_env = load_config(model_dir/"config_env.yaml")
        config_rl = load_config(model_dir/"config_env.yaml")
        config_ft = load_config(config_exp["ft"])

        env = build_env(config_env, seed=args.seed)

        print("\nCheck environment consistency...")
        check_env(env, warn=True)
        print("Done")

        model = load_model(env, config_ft, load_path=config_exp["model"], seed=args.seed)

        run_dir, tb_dir = create_run(config_exp)

        env = Monitor(env, str(run_dir))
        model.tensorboard_log = str(tb_dir)

        model = train(model, config_ft)

        save_model(model, run_dir)
        save_config(config_exp, run_dir, "config_exp.yaml")
        save_config(config_env, run_dir, "config_env.yaml")
        save_config(config_rl, run_dir, "config_rl.yaml")
        save_config(config_ft, run_dir, "config_ft.yaml")

    else:
        parser.error("Provide a consistent config: either train or finetuning config")



if __name__ == "__main__":
    main()

