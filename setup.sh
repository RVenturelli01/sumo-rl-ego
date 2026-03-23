#!/usr/bin/env bash

set -euo pipefail

MODE="${1:---full}"
VENV_DIR=".venv"

case "$MODE" in
  --minimal)
    INSTALL_TARGET="."
    ;;
  --full)
    INSTALL_TARGET=".[experiments,dev]"
    ;;
  *)
    echo "Usage: ./setup.sh [--minimal|--full]"
    exit 1
    ;;
esac

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "$INSTALL_TARGET"

if command -v sumo >/dev/null 2>&1; then
  echo "SUMO binary found: $(command -v sumo)"
else
  echo "Warning: 'sumo' was not found on PATH."
  echo "Install SUMO and make sure 'sumo' is available before running the environments."
fi

echo
echo "Setup complete."
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo "Minimal package smoke test:"
echo "  python -c \"import sumo_rl_ego as sre; print(sre.list_envs())\""
