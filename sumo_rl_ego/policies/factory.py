from .model_policy import ModelPolicy
from .registry import POLICY_REGISTRY, list_models, _model_and_config_path
from omegaconf import OmegaConf
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

ALGO_REGISTRY = {
    "PPO": PPO,
    "DQN": DQN,
    "A2C": A2C,
    "SAC": SAC,
    "TD3": TD3,
}

def load_policy(policy_id, env=None):
    # 1. Policy registry
    if policy_id in POLICY_REGISTRY:
        return POLICY_REGISTRY[policy_id]()

    # 2. Model registry
    if policy_id in list_models():
        model_path, cfg_path = _model_and_config_path(policy_id)

        cfg = OmegaConf.load(cfg_path)
        algo_cls = ALGO_REGISTRY[cfg.model.algo]
        
        model = algo_cls.load(model_path, env=env, device="cpu")
        return ModelPolicy(model)

    raise ValueError(f"Unknown policy '{policy_id}'")
