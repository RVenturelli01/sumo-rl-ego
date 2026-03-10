import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import sumo_rl_ego as sre


@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    if cfg.model_path is not None:
        cfg_old = sre.load_run(cfg.model_path)
        env = sre.make_env(cfg_old.env, seed=cfg.seed)
        model = sre.load_model(env=env, cfg=cfg_old.algo, seed=cfg.seed, load_path=cfg.model_path)
        policy = sre.policies.SB3Policy(model)

    elif cfg.policy_id is not None:
        env = sre.make_env(cfg.env, seed=cfg.seed)
        policy = sre.load_policy(cfg.policy_id)

    else:
        raise ValueError("Either model_path or policy_id must be provided.")


    pbar = tqdm(total=cfg.episodes, desc="Episodes")

    for ep in range(cfg.episodes):

        obs, _ = env.reset(seed=cfg.seed + ep)

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

        pbar.update(1)

    pbar.close()
    env.close()

    log = env.metrics_tracker.get_log_metrics()
    print_log(log)



def print_log(log):
    groups = {}

    for key, value in log.items():
        group, name = key.split("/", 1)

        if group not in groups:
            groups[group] = {}

        groups[group][name] = value

    for group, items in groups.items():
        print(f"[{group}]")
        for name, value in items.items():
            if isinstance(value, float):
                value = round(value, 3)
            print(f"  {name:20s} : {value}")
        print()


if __name__ == "__main__":
    main()