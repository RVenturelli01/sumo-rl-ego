from datetime import datetime
from pathlib import Path
from sumo_rl_ego.infra.utils.project import find_repo_root


def create_run(config):

    repo_root = find_repo_root()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = config.get("meta", {}).get("name", "experiment")

    run_name = f"{timestamp}_{name}"

    run_dir = repo_root / "outputs" / run_name
    tb_dir = run_dir / "tensorboard"

    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, tb_dir