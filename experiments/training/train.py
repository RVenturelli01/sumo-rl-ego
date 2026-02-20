from pathlib import Path

from sumo_rl_ego.env.config import SumoConfig
from sumo_rl_ego.env.sumo_env import SumoEnv

from experiments.modules.my_callback import MyCallback
from experiments.modules.my_ego import MyEgo
from experiments.modules.my_observation import MyObservation
from experiments.modules.my_reward import MyReward
from experiments.modules.my_metrics import MyMetrics

from rl_utils.config_loader import load_rl_config
from rl_utils.model_factory import build_model
from rl_utils.trainer import train
from stable_baselines3.common.env_checker import check_env

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(__file__).resolve().parent.parent.parent


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
              metrics_tracker=MyMetrics(window=100))

# Optional sanity check
check_env(env, warn=True)

# Build model
model = build_model(env, cfg)

# Train
train(model, cfg, env, root=REPO_DIR, callback=MyCallback)


