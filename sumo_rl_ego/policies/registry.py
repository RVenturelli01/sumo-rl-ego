import os
from os import path
from pathlib import Path


POLICY_REGISTRY = {}


def register_policy(name):
    def decorator(cls):
        POLICY_REGISTRY[name] = cls
        return cls
    return decorator


def list_policies():
    return sorted(POLICY_REGISTRY.keys())




MODELS_DIR = Path(__file__).parent / "models"


def list_models():
    return [
        d for d in os.listdir(MODELS_DIR)
        if os.path.isdir(MODELS_DIR / d)
    ]

def _model_and_config_path(policy_id):
    model_path = MODELS_DIR / policy_id / "model.zip"
    cfg_path = MODELS_DIR / policy_id / "config.yaml"

    return model_path, cfg_path



