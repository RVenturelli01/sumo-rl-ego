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


def load_config_from_model(model_path: str) -> dict:
    """
    Load config stored alongside a trained model.
    Used for resume or evaluation.
    """
    print("\n[INFRA] Loading config from model folder...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[INFRA] Model not found: {model_path}")

    model_dir = os.path.dirname(model_path)
    saved_cfg_path = os.path.join(model_dir, "config.yaml")

    if not os.path.exists(saved_cfg_path):
        raise FileNotFoundError(
            f"[INFRA] No config.yaml found next to model: {saved_cfg_path}"
        )

    with open(saved_cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[INFRA] Config loaded from model folder: {saved_cfg_path}")
    return cfg