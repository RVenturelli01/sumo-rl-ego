from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..policies.factory import policy_from_model


def _reset_env(env, seed=None):
    if seed is None:
        return env.reset()
    return env.reset(seed=seed)


def _start_episode(env, policy, seed=None):
    policy = policy_from_model(policy)
    reset_result = _reset_env(env, seed=seed)

    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    policy.reset()
    return policy, obs


def run_episode(env, policy, seed=None, max_steps=None):
    policy, obs = _start_episode(env, policy, seed=seed)

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    return info
