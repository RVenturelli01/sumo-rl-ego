import os
import yaml


def load_config(config_path: str | None = None,
                model_path: str | None = None) -> dict:

    cfg = None

    # --- 1. Try loading from model folder (eval mode or fine-train mode) ---
    if model_path:
        model_dir = os.path.dirname(model_path)
        saved_cfg_path = os.path.join(model_dir, "config.yaml")
        print(saved_cfg_path)

        if os.path.exists(saved_cfg_path):
            with open(saved_cfg_path) as f:
                cfg = yaml.safe_load(f)
                print(f"Config loaded from model folder: {saved_cfg_path}")

    # --- 2. Fallback to CLI config (train mode) ---
    if cfg is None and config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            print(f"Config loaded from CLI path: {config_path}")

    # --- 3. Hard fail if nothing found ---
    if cfg is None:
        raise ValueError(
            "No config found. Provide --config for training "
            "or ensure config.yaml exists in model folder."
        )

    return cfg