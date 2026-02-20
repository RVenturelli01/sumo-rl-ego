from pathlib import Path
import pprint
from tqdm import tqdm


import traci
import numpy as np
from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from experiments.modules.my_ego import MyEgo
from experiments.modules.my_observation import MyObservation
from experiments.modules.my_reward import MyReward
from experiments.modules.my_metrics import MyMetrics

from rl_utils.config_loader import load_rl_config
from experiments.policies.sb3_policy import SB3Policy
from stable_baselines3.common.env_checker import check_env

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = REPO_DIR / "models" / "test_dqn_highway_2026-02-20_18-43-13" / "model.zip"


# Load YAML config
CFG_PATH = SCRIPT_DIR.parent / "configs" / "dqn.yaml"
cfg = load_rl_config(CFG_PATH)

# Build SumoConfig
sumo_cfg = SumoConfig(**cfg["sumo_config"])

# Build environment
env = SumoEnv(sumo_cfg, 
              ego_controller=MyEgo(), 
              obs_builder=MyObservation(), 
              reward_function=MyReward(), 
              metrics_tracker=MyMetrics())

# get the trained model
model = SB3Policy(model_path=MODEL_PATH, algo=cfg["algorithm"])

# bar for tracking episode progress
N_iterations = 100
pbar = tqdm(total=N_iterations, desc="Episodes")

# Rollout loop
for i in range(N_iterations):
    pbar.update(1)

    # start the environment and get the first observation
    obs, _ = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

pbar.close()
env.close()

global_metrics = env.metrics_tracker.get_global_metrics()
pprint.pprint(global_metrics)              
