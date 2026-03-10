from pathlib import Path
from omegaconf import OmegaConf


def load_run(model_path):

    model_path = Path(model_path)

    run_dir = model_path.parent
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    return cfg