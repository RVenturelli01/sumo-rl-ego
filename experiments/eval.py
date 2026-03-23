import numpy as np
import hydra
import wandb
import sumo_rl_ego as sre

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf

from experiments.config_utils import (
    init_wandb, 
    resolve_paths,
    print_eval_cfg,
    confirm_cfg,
    check_source_cfg,
    load_policy_from_cfg,
)
    

class EvalMetrics:

    def __init__(self):
        self.data = defaultdict(list)

        # choose here
        self.with_std = {
            "performance/speed_mean",
            "performance/return",
            "performance/length",
            "performance/duration",
        }

    def add_episode(self, info):
        ep = info.get("metrics", {}).get("episode", {})

        # --- performance ---
        self.data["performance/speed_mean"].append(ep.get("performance/speed_mean", 0.0))
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

    def compute(self):
        out = {}
        for k, v in self.data.items():
            if not v:
                continue

            out[k + "_mean"] = np.mean(v)

            if k in self.with_std:
                out[k + "_std"] = np.std(v)

        return out
    
    
def print_metrics(d):
    current = ""

    for k, v in d.items():
        if not k.endswith("_mean"):
            continue

        sec, name = k.split("/")
        name = name.replace("_mean", "")

        if sec != current:
            current = sec
            print(f"\n=== {sec} ===")

        std_key = k.replace("_mean", "_std")

        if std_key in d:
            print(f"{name}: {float(v):.3f} ± {float(d[std_key]):.3f}")
        else:
            print(f"{name}: {float(v):.3f}")

from pathlib import Path
from hydra.utils import get_original_cwd


@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    resolve_paths(cfg)
    check_source_cfg(cfg)

    print_eval_cfg(cfg, "EVAL")
    confirm_cfg()


    run = init_wandb(cfg)
    env = None

    try:
        print("Loading environment...")
        env = sre.make_env(
            cfg.env.id, 
            seed=cfg.seed, 
            **cfg.env.kwargs
        )

        print("Loading policy...")
        policy = load_policy_from_cfg(cfg, env=env)

        metrics = EvalMetrics()

        print("Running evaluation...")
        for _ in range(cfg.n_episodes):

            info = sre.run_episode(
                env,
                policy,
                seed=cfg.seed,
            )
            
            metrics.add_episode(info)

        results = metrics.compute()

        if run is not None:
            wandb.log(results)

        print_metrics(results)

    finally:
        if env is not None:
            env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
