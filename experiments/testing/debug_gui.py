from pathlib import Path

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
MODEL_PATH = REPO_DIR / "models" / "test_dqn_highway_2026-02-19_18-43-14" / "model.zip"


# Load YAML config
CFG_PATH = SCRIPT_DIR.parent / "configs" / "dqn.yaml"
cfg = load_rl_config(CFG_PATH)

# Build SumoConfig
sumo_cfg = SumoConfig(**cfg["sumo_config"], use_gui=True)

# Build environment
env = SumoEnv(sumo_cfg, 
              ego_controller=MyEgo(), 
              obs_builder=MyObservation(), 
              reward_function=MyReward(), 
              metrics_tracker=MyMetrics())

# get the trained model
model = SB3Policy(model_path=MODEL_PATH, algo=cfg["algorithm"])

# start the environment and get the first observation
obs, _ = env.reset()

# opzionale: fai seguire l'ego
traci.gui.trackVehicle("View #0", "ego")
traci.gui.setZoom("View #0", 2000)
input("Premi invio per chiudere...") 

# Rollout loop
while True:

    # deterministic=True = niente epsilon-greedy
    print(obs)
    action = model.predict(obs)

    print("="*50)
    print(f"Action: {env.ego_controller.print_action(action)}")
    print("-"*50)
    print(env.obs_builder.print_obs(obs))

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode finished:", info)
        obs, _ = env.reset()
        traci.gui.trackVehicle("View #0", "ego")
        input("Premi invio per chiudere...") 

env.close()
