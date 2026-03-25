import numpy as np
import hydra
import wandb
import sumo_rl_ego as sre

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from sumo_rl_ego.utils import (
    init_wandb, 
    resolve_paths,
    confirm_cfg,
    check_source_cfg,
    load_policy_from_cfg,
)
    

class EvalMetrics:

    def __init__(self):
        self.data = defaultdict(list)

        # choose here
        self.with_std = {
            "performance/avg_speed",
            "performance/return",
            "performance/length",
            "performance/duration",
        }

    def add_episode(self, info):
        ep = info.get("metrics", {}).get("episode", {})

        # --- performance ---
        self.data["performance/avg_speed"].append(ep.get("performance/speed_mean", 0.0))
        self.data["performance/return"].append(ep.get("performance/return", 0.0))
        self.data["performance/length"].append(info.get("step", 0))
        self.data["performance/duration"].append(info.get("time", 0.0))

        # --- actions ---
        self.data["action_rate/ss"].append(ep.get("action_rate/ss", 0.0))
        self.data["action_rate/lcl"].append(ep.get("action_rate/lcl", 0.0))
        self.data["action_rate/lcr"].append(ep.get("action_rate/lcr", 0.0))
        self.data["action_rate/acc"].append(ep.get("action_rate/acc", 0.0))
        self.data["action_rate/dec"].append(ep.get("action_rate/dec", 0.0))

        # --- events ---
        event = info.get("event", "running")
        self.data["event_rate/collisions"].append(int(event == "collided"))
        self.data["event_rate/off_road"].append(int(event == "off_road"))
        self.data["event_rate/timeouts"].append(int(event == "timeout"))
        self.data["event_rate/successes"].append(int(event == "arrived"))


    def print_metrics(self):
        current = ""

        for k, v in self.data.items():
            if not v:
                continue

            mean = np.mean(v)
            std = np.std(v) if k in self.with_std else None

            sec, name = k.split("/")
            name = name

            if sec != current:
                current = sec
                print(f"\n=== {sec} ===")

            if std is not None:
                print(f"{name}: {mean:.3f} ± {std:.3f}")
            else:
                print(f"{name}: {mean:.3f}")


    def log_metrics(self, run):

        log_histogram(
            data=self.data["performance/duration"],
            value="duration",
            title="Time Distribution")

        log_histogram(
            data=self.data["performance/return"],
            value="return",
            title="Return Distribution")

        log_histogram(
            data=self.data["performance/avg_speed"],
            value="avg_speed",
            title="Average Speed Distribution")
        
        log_bar_plot(
            data=[
                ["collisions", np.mean(self.data["event_rate/collisions"])],
                ["off_road", np.mean(self.data["event_rate/off_road"])],
                ["timeouts", np.mean(self.data["event_rate/timeouts"])],
                ["successes", np.mean(self.data["event_rate/successes"])],
            ],
            value="rate",
            title="Event Rates",
        )


def log_histogram(data, value, title):
    data=[[i, float(v)] for i, v in enumerate(data)]

    table = wandb.Table(
        data=data,
        columns=["x", value],
    )
    histogram = wandb.plot.histogram(
        table,
        value=value,
        title=title,
    )

    print(f"Logging histogram: {title}")
    wandb.log({title: histogram})
    

def log_bar_plot(data, value, title):
    
    table = wandb.Table(
        data=data,
        columns=["class", value],
    )

    bar_plot = wandb.plot.bar(
        table,
        label="class",        
        value=value,   
        title=title,
    )

    print(f"Logging bar plot: {title}")
    wandb.log({title: bar_plot})


def print_eval_cfg(cfg):
    print(f"\n========== EVAL CONFIG ==========\n")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("================== Summary ==================\n")
    print(f"Environment: {cfg.env.id}")
    print(f"Environment arguments: {cfg.env.kwargs}")

    # Print all non-null source fields
    for key, value in cfg.source.items():
        if value is not None:
            if key == "policy_class" and value.get("__target__") is None:
                continue
            print(f"{key}: {value}")

    print(f"Number of episodes: {cfg.run.n_episodes}")
    print("\n=============================================\n")





@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    resolve_paths(cfg)
    check_source_cfg(cfg)

    print_eval_cfg(cfg)
    confirm_cfg()


    run = init_wandb(cfg)
    env = None

    try:
        print("Loading environment...")
        env = sre.make_env(
            cfg.env.id, 
            seed=cfg.run.seed, 
            **cfg.env.kwargs
        )

        print("Loading policy...")
        policy = load_policy_from_cfg(cfg, env=env)

        metrics = EvalMetrics()

        print("Running evaluation...")
        for _ in range(cfg.run.n_episodes):

            info = sre.run_episode(
                env,
                policy,
                seed=cfg.run.seed,
            )
            
            metrics.add_episode(info)

        if run is not None:
            metrics.log_metrics(run)

        metrics.print_metrics()

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
