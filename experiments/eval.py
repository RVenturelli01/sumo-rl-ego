import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

import sumo_rl_ego as sre


'''
Evaluation script for trained models or hand-crafted policies.

it plots:
- episode rewards over episodes
- episode lengths over episodes
- episode average speeds over episodes
- histogram of the episode rewards
- histogram of the episode lengths
- histogram of the episode average speeds
'''



@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=dict(cfg)
    )

    if cfg.model_path is not None:
        cfg_old = sre.load_run(cfg.model_path)
        env = sre.make_env(cfg_old.env, seed=cfg.seed)
        model = sre.load_model(
            env=env,
            cfg=cfg_old.algo,
            seed=cfg.seed,
            load_path=cfg.model_path
        )
        policy = sre.policies.SB3Policy(model)

    elif cfg.policy_id is not None:
        env = sre.make_env(cfg.env, seed=cfg.seed)
        policy = sre.load_policy(cfg.policy_id)

    else:
        raise ValueError("Either model_path or policy_id must be provided.")


    pbar = tqdm(total=cfg.episodes, desc="Episodes")

    history_reward = []
    history_length = []
    history_avg_speed = []

    for ep in range(cfg.episodes):

        seed=cfg.seed + ep
        obs, _ = env.reset(seed)

        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action = policy.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        ep_length = info["metrics"]["episode"]["ep_length"]
        ep_avg_speed = info["metrics"]["episode"]["ep_avg_speed"]

        history_reward.append(ep_reward)
        history_length.append(ep_length)
        history_avg_speed.append(ep_avg_speed)

        wandb.log(
            {
                "episode_reward": ep_reward,
                "episode_length": ep_length,
                "episode_avg_speed": ep_avg_speed,
            },
            step=seed,
        )

        pbar.update(1)

    pbar.close()
    env.close()


    log = env.metrics_tracker.get_log_metrics()
    print_log(log)


    # histogram of the returns on wandb
    wandb.log(
        {
            "returns_histogram": build_histogram_plot(
                history_reward,
                column="episode_reward",
                title="Episode Reward Histogram",
            ),
            "lengths_histogram": build_histogram_plot(
                history_length,
                column="episode_length",
                title="Episode Length Histogram",
            ),
            "avg_speed_histogram": build_histogram_plot(
                history_avg_speed,
                column="episode_avg_speed",
                title="Episode Average Speed Histogram",
            ),
        }
    )


    wandb.finish()





def build_histogram_plot(values, column, title):
    histogram_table = wandb.Table(
        data=[[value] for value in values],
        columns=[column],
    )
    return wandb.plot.histogram(
        histogram_table,
        value=column,
        title=title,
    )




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