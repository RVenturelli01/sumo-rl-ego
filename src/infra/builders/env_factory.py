from sumo_rl_ego.env.sumo_env import SumoEnv
from sumo_rl_ego.env.config import SumoConfig
from src.infra.utils.class_loader import build_class



def build_env(cfg) -> SumoEnv:

    sumo_cfg = SumoConfig(**cfg["sumo_config"])

    ego_class = build_class(cfg["env"]["ego"]["class"], args=cfg["env"]["ego"]["args"])
    obs_class = build_class(cfg["env"]["obs"]["class"], args=cfg["env"]["obs"]["args"])
    reward_class = build_class(cfg["env"]["reward"]["class"], args=cfg["env"]["reward"]["args"])
    metrics_class = build_class(cfg["env"]["metrics"]["class"], args=cfg["env"]["metrics"]["args"])

    env = SumoEnv(sumo_cfg,
                ego_controller=ego_class,
                obs_builder=obs_class,
                reward_function=reward_class,
                metrics_tracker=metrics_class)  
    
    print("Env loaded with the following plugins:")
    print(cfg["env"]["ego"]["class"])
    print(cfg["env"]["obs"]["class"])
    print(cfg["env"]["reward"]["class"])
    print(cfg["env"]["metrics"]["class"])

    return env