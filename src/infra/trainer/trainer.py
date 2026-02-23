import yaml
from datetime import datetime
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from src.infra.utils.project import find_repo_root
from src.infra.utils.costum_logs_callback import CostumLogsCallback


def train(model, env, cfg):
    # --- Repo root ---
    REPO_ROOT = find_repo_root()

    # --- Nome run ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    custom_name = cfg.get("meta", {}).get("name")
    algo = cfg["algorithm"].lower()
    run_name = f"{timestamp}_{custom_name}" if custom_name else f"{timestamp}_{algo}"

    RUN_DIR = REPO_ROOT / "outputs" / run_name
    TB_DIR = RUN_DIR / "tensorboard"

    # crea cartelle
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)

    # --- Monitor SB3 (salva monitor.csv nei logs) ---
    env = Monitor(env, str(RUN_DIR))

    # --- TensorBoard ---
    model.tensorboard_log = str(TB_DIR)

    # --- Parametri training ---
    learn_kwargs = cfg.get("training", {}).copy()
    if "total_timesteps" not in learn_kwargs:
        raise ValueError("Config must contain training.total_timesteps")

    total_steps = learn_kwargs.pop("total_timesteps")

    # --- Training ---
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        callback=CostumLogsCallback(),  
        tb_log_name="train",
        **learn_kwargs,
    )

    # --- Salvataggio modello ---
    model_path = RUN_DIR / "model"
    model.save(model_path)

    # --- Salvataggio config ---
    config_path = RUN_DIR / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"\nModel saved: {model_path}.zip")
    print(f"Config saved: {config_path}")
    print(f"TensorBoard: tensorboard --logdir outputs")