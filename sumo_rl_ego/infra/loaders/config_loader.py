import os
import yaml


def load_config(config_path: str) -> dict:
    """
    Load config directly from a YAML file.
    Used for training from scratch or fine-tuning.
    """
    print("\n[INFRA] Loading config from YAML...")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[INFRA] Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[INFRA] Config loaded from: {config_path}")
    return cfg

