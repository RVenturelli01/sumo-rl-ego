import yaml


def save_model(model, run_dir):
    model_path = run_dir / "model"
    model.save(model_path)
    print(f"\nModel saved: {model_path}.zip")


def save_config(config, run_dir, name):
    config_path = run_dir / name

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"Config saved: {config_path}")