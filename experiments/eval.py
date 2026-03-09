import os

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.policy.sb3_policy import SB3Policy


@hydra.main(version_base=None, config_path="configs", config_name="exp/eval")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    env = build_env(cfg.env, seed=cfg.seed)

    if cfg.model_path:
        model = load_model(env, cfg.rl, load_path=cfg.model_path, seed=cfg.seed)
        policy = SB3Policy(model=model)

    elif cfg.policy._target_:
        policy = hydra.utils.instantiate(cfg.policy)

    else:
        raise ValueError("You must provide either model_path or policy._target_")

    pbar = tqdm(total=cfg.episodes, desc="Episodes")

    for ep in range(cfg.episodes):

        obs, _ = env.reset(seed=cfg.seed + ep)

        terminated = False
        truncated = False

        while not terminated and not truncated:

            action = policy.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)

        pbar.update(1)

    pbar.close()
    env.close()

    env.metrics_tracker.print_log_metrics()


if __name__ == "__main__":
    main()