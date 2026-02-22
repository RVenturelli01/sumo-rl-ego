from pathlib import Path

def find_repo_root(start: Path = None, markers=("pyproject.toml", "README.md")):
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        if any((parent / m).exists() for m in markers):
            return parent

    raise RuntimeError("Repo root not found")