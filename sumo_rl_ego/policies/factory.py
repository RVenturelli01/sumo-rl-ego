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

def load_policy(policy_id: str, env=None) -> "Policy":
    """Load a policy by ID.

    Lookup order:
      1. **Policy registry** — rule-based policies registered with @register_policy
         (e.g. ``"FastPolicy-v0"``). Use ``list_policies()`` to see available IDs.
      2. **Model registry** — pre-trained SB3 models stored in
         ``sumo_rl_ego/policies/models/``. Use ``list_models()`` to see available IDs.

    Args:
        policy_id: ID string as returned by ``list_policies()`` or ``list_models()``.
        env: Optional Gymnasium environment passed to SB3 ``model.load()``.
             Required when loading a model that needs observation/action space info.

    Returns:
        A :class:`Policy` instance ready to call ``predict(obs)``.

    Raises:
        ValueError: If ``policy_id`` is not found in either registry.

    Examples:
        >>> policy = load_policy("FastPolicy-v0")
        >>> policy = load_policy("dqn_highway_v1", env=env)
    """
    # 1. Policy registry (rule-based / hand-crafted)
    if policy_id in POLICY_REGISTRY:
        return POLICY_REGISTRY[policy_id]()

    # 2. Model registry (pre-trained SB3 checkpoints)
    if policy_id in list_models():
        model_path, cfg_path = _model_and_config_path(policy_id)

        cfg = OmegaConf.load(cfg_path)
        algo_cls = ALGO_REGISTRY[cfg.model.algo]

        model = algo_cls.load(model_path, env=env, device="cpu")
        return ModelPolicy(model)

    available = sorted(list(POLICY_REGISTRY) + list_models())
    raise ValueError(
        f"Unknown policy '{policy_id}'. Available: {available}"
    )
