import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import traci
from stable_baselines3 import PPO, DQN, A2C, SAC, TD3

import sumo_rl_ego as sre


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

    print("\n=======================================\n")



def load_policy_and_env(cfg):
    
    if cfg.mode == "model":
        cfg_old = sre.load_run(cfg.model_path)
        exp_old = cfg_old.experiment

        env = sre.make_env(exp_old.env_id, seed=cfg.seed)

        algo_cls = ALGO_REGISTRY[exp_old.algo]

        model = algo_cls.load(
            env=env,
            load_path=cfg.model_path,
            seed=cfg.seed,
        )

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


def manual_rollout(env, policy, seed):

    obs, _ = env.reset(seed=seed)

    while True:

        action = policy.predict(obs)

        print("=" * 20 + "ACTION" + "=" * 20)
        env.ego_controller.print_action(action)

        print("=" * 20 + "OBSERVATION" + "=" * 20)
        env.obs_builder.print_obs(obs)
        print("=" * 50)

        input("Press Enter to step...\n")

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished:", info.get("event"))
            input("Press Enter to reset...\n")
            obs, _ = env.reset()


def auto_rollout(env, policy, cfg):

    obs, _ = env.reset(seed=cfg.seed)

    while True:

        action = policy.predict(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Episode finished:", info.get("event"))
            obs, _ = env.reset()


def choose_rollout_mode(default_manual: bool):

    default = "manual" if default_manual else "auto"

    answer = input(
        f"\nRollout mode? [m=manual, a=auto, Enter={default}]: "
    ).strip().lower()

    if answer == "":
        return default_manual

    if answer in ["m", "manual"]:
        return True

    if answer in ["a", "auto"]:
        return False

    print("Invalid input, using default.")
    return default_manual



@hydra.main(version_base=None, config_path="configs", config_name="play.yaml")
def main(cfg: DictConfig):

    print_eval_config(cfg)
    confirm_start()

    print("Loading policy and environment...")
    policy, env = load_policy_and_env(cfg)

    print("Starting play...")

    manual = choose_rollout_mode(cfg.manual_rollout)

    if manual:
        manual_rollout(env, policy, cfg.seed)
    else:
        auto_rollout(env, policy, cfg)

    env.close()



if __name__ == "__main__":
    main()