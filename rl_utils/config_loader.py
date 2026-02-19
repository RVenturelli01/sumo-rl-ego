import yaml
from pathlib import Path

def load_rl_config(path):
    path = Path(path).resolve()
    print(f"[RL CONFIG] Cerco file in: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)
