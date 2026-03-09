from pathlib import Path

def find_repo_root(start: Path = None, markers=("pyproject.toml", "sumo_rl_ego")):
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        if all((parent / m).exists() for m in markers):
            return parent

    raise RuntimeError("Repo root not found")