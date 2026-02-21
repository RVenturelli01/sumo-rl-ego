from pathlib import Path
from datetime import datetime
import yaml
from stable_baselines3.common.monitor import Monitor

def train(model, cfg: dict, env, root: Path = Path("."), callback=None):
    """
    Train SB3 model with clean structure:

    models/<experiment>/<run>/
    logs/<experiment>/<run>/

    Saves:
    - model.zip (models)
    - config.yaml (models)
    - tensorboard (logs)
    - monitor.csv (logs)
    """

    root = Path(root)

    models_root = root / "models"
    logs_root = root / "logs" 

    # ===============================
    # Run naming
    # ===============================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    custom_name = cfg.get("meta", {}).get("name")

    algo = cfg["algorithm"].lower()
    run_name = f"{custom_name}_{timestamp}" if custom_name else f"{algo}_{timestamp}"

    model_dir = models_root / run_name
    log_dir = logs_root / run_name

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RL] Run: {run_name}")
    print(f"[RL] Models: {model_dir}")
    print(f"[RL] Logs:   {log_dir}")

    # ===============================
    # Monitor (CSV logs)
    # ===============================
    # env = Monitor(env, log_dir)

    # ===============================
    # TensorBoard
    # ===============================
    tb_dir = log_dir / "tensorboard"
    model.tensorboard_log = str(tb_dir)

    # ===============================
    # Training params
    # ===============================
    learn_kwargs = cfg.get("training", {}).copy()

    if "total_timesteps" not in learn_kwargs:
        raise ValueError("Config must contain training.total_timesteps")

    total_steps = learn_kwargs.pop("total_timesteps")

    print(f"[RL] Algorithm: {cfg['algorithm']}")
    print(f"[RL] Timesteps: {total_steps}")

    # ===============================
    # Train
    # ===============================
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        callback=callback,
        tb_log_name="sumo_rl",
        **learn_kwargs,
    )

    # ===============================
    # Save model (models/)
    # ===============================
    model_path = model_dir / "model"
    model.save(model_path)

    # ===============================
    # Save config next to model
    # ===============================
    config_path = model_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print(f"\n[RL] Model saved: {model_path}.zip")
    print(f"[RL] Config saved: {config_path}")
    print(f"[RL] TensorBoard: tensorboard --logdir logs")

    return model_dir, log_dir
