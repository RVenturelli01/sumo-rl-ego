from .registry import POLICY_REGISTRY

def load_policy(policy_id):

    # rule-based
    if policy_id in POLICY_REGISTRY:
        return POLICY_REGISTRY[policy_id]()

    raise ValueError(f"Unknown policy '{policy_id}'")
