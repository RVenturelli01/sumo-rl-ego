import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3

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

ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}



def confirm_start():
    answer = input("\nStart training? [y/N]: ").strip().lower()
    if answer not in ["y", "yes"]:
        print("Training aborted by user.")
        exit()



def print_eval_config(cfg):

    print("\n========== EVALUATION CONFIG ==========\n")

    print(f"n_episodes: {cfg.n_episodes}")
    print(f"seed: {cfg.seed}")
    print(f"mode: {cfg.mode}")

    if cfg.mode == "model":
        print(f"model_path: {cfg.model_path}")

    elif cfg.mode == "policy_id":
        print(f"policy_id: {cfg.policy_id}")
        print(f"env_id: {cfg.env_id}")

    elif cfg.mode == "policy_class":
        print(f"policy_class: {cfg.policy_class}")
        print(f"env_id: {cfg.env_id}")

    print("\nwandb:")
    print(f"  project: {cfg.wandb.project}")
    print(f"  run_name: {cfg.wandb.run_name}")

    print("\n=======================================\n")



def run_episode(env, policy, seed):

    obs, _ = env.reset(seed=seed)

    terminated = truncated = False
    ep_reward = 0.0

    while not (terminated or truncated):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

    metrics = info["metrics"]["episode"]

    return ep_reward, metrics["ep_length"], metrics["ep_avg_speed"]



def load_policy_and_env(cfg):
    
    if cfg.mode == "model":
        cfg_old = sre.load_run(cfg.model_path)
        exp_old = cfg_old.experiment

        env = sre.make_env(exp_old.env_id, seed=cfg.seed)

        algo_cls = ALGO_REGISTRY[exp_old.algo]

        model = algo_cls.load(cfg.model_path, env=env)

        policy = sre.policies.SB3Policy(model)

    elif cfg.mode == "policy_id":
        env = sre.make_env(cfg.env_id, seed=cfg.seed)
        policy = sre.load_policy(cfg.policy_id)

    elif cfg.mode == "policy_class":
        env = sre.make_env(cfg.env_id, seed=cfg.seed)
        policy = instantiate(cfg.policy_class, env=env)
        
    else:
        raise ValueError("Invalid mode. Must be one of 'model', 'policy_id', or 'policy_class'.")

    return policy, env


@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):

    print_eval_config(cfg)
    confirm_start()
    
    print("Initializing Weights & Biases...")
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
        sync_tensorboard=cfg.wandb.sync_tensorboard,
        dir="/tmp"
    )

    print("Loading policy...")

    policy, env = load_policy_and_env(cfg)

    print("Evaluating policy...")

    pbar = tqdm(total=cfg.n_episodes, desc="Episodes")

    history_reward = []
    history_length = []
    history_avg_speed = []

    for ep in range(cfg.n_episodes):

        seed = cfg.seed + ep

        ep_reward, ep_length, ep_avg_speed = run_episode(env, policy, seed)

        history_reward.append(ep_reward)
        history_length.append(ep_length)
        history_avg_speed.append(ep_avg_speed)

        wandb.log({
                "episode_reward": ep_reward,
                "episode_length": ep_length,
                "episode_avg_speed": ep_avg_speed,
            },step=seed,
        )

        pbar.update(1)

    pbar.close()
    env.close()


    log = env.metrics_tracker.get_log_metrics()
    print_log(log)


    wandb.log({
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
            ),}
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