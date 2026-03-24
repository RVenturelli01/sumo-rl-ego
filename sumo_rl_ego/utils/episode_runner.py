from __future__ import annotations

def run_episode(env, policy, seed=None):
    obs, _ = env.reset(seed=seed)
    policy.reset()

    terminated = False
    truncated = False
    info = {}

    while not (terminated or truncated):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    return info
