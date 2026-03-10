from stable_baselines3.common.vec_env import SubprocVecEnv
from .registry import ENV_REGISTRY



def make_env(env_id: str, seed: int, **kwargs):
    print("[SRE] Creating environment...")

    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_id '{env_id}'")

    env_cls = ENV_REGISTRY[env_id]

    return env_cls(seed=seed, **kwargs)



def make_vec_env(env_id: str, n_envs: int, base_seed: int = 0, **kwargs):
    print("[SRE] Creating vectorized environment...")

    def make_thunk(rank: int):
        def _init():
            seed = base_seed + rank
            env = make_env(env_id, seed=seed, **kwargs)
            return env
        return _init

    env = SubprocVecEnv([make_thunk(i) for i in range(n_envs)])

    return env
