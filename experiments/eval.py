import argparse
from tqdm import tqdm
import pprint

from src.infra.loaders.config_loader import load_config
from src.infra.builders.env_factory import build_env
from src.infra.builders.model_factory import load_model

from src.infra.policy.sb3_policy import SB3Policy
from src.infra.loaders.class_loader import build_class


DEFAULT_MODEL = None # "outputs/models/test_dqn_highway_2026-02-21_22-43-05/model.zip"
DEFAULT_CONFIG = "experiments/configs/dqn.yaml"
DEFAULT_EPISODES = 20



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_policy", required=False, default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=False, default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config_policy, args.model)

    # Build env (incapsula SumoConfig + builders vari)
    env = build_env(cfg)

    # Load trained model (SB3Policy wrapper gestito in model_factory)
    if args.model:
        model = load_model(env, cfg, load_path=args.model)
        policy = SB3Policy(model=model)
    elif args.config_policy:
        policy = build_class(cfg["policy"]["class"], cfg["policy"]["args"])
        
    # Rollout loop
    pbar = tqdm(total=args.episodes, desc="Episodes")

    for _ in range(args.episodes):
        pbar.update(1)

        obs, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

    pbar.close()
    env.close()

    # Global metrics
    global_metrics = env.metrics_tracker.get_rollout_metrics()
    pprint.pprint(global_metrics)


if __name__ == "__main__":
    main()