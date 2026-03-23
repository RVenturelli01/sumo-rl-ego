from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from ..policies.factory import policy_from_model
from .episode_runner import _start_episode, run_episode


def play_policy(env, policy, seed=0, manual=False, max_episodes=1):
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")

    policy = policy_from_model(policy)
    results = []
    episode_idx = 0

    while max_episodes is None or episode_idx < max_episodes:
        episode_seed = None if seed is None else seed + episode_idx
        policy, obs = _start_episode(env, policy, seed=episode_seed)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action = policy.predict(obs)

            if manual:
                print("=" * 20 + "ACTION" + "=" * 20)
                if hasattr(env, "ego_controller") and hasattr(env.ego_controller, "print_action"):
                    env.ego_controller.print_action(action)
                else:
                    print(action)

                print("=" * 20 + "OBSERVATION" + "=" * 20)
                if hasattr(env, "obs_builder") and hasattr(env.obs_builder, "print_obs"):
                    env.obs_builder.print_obs(obs)
                else:
                    print(obs)

                print("=" * 50)
                input("Press Enter to step...\n")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        episode_log = dict(info.get("log", {}))
        episode_log.setdefault("episode/return", total_reward)
        info["log"] = episode_log
        info.setdefault("step", steps)
        results.append(info)

        print("Episode finished:", info.get("event"))

        if manual and (max_episodes is None or episode_idx + 1 < max_episodes):
            input("Press Enter to reset...\n")

        episode_idx += 1

    return results
