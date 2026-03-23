from .base_policy import Policy
from .model_policy import ModelPolicy
from .registry import POLICY_REGISTRY


def policy_from_model(model_or_policy):
    if isinstance(model_or_policy, Policy):
        return model_or_policy

    return ModelPolicy(model_or_policy)


def load_policy(policy_id):
    if policy_id in POLICY_REGISTRY:
        return POLICY_REGISTRY[policy_id]()

    raise ValueError(f"Unknown policy '{policy_id}'")
