import hydra
from omegaconf import DictConfig, OmegaConf

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

from sumo_rl_ego.infra.builders.env_factory import build_env
from sumo_rl_ego.infra.builders.model_factory import load_model
from sumo_rl_ego.infra.trainer.trainer import train


@hydra.main(version_base=None, config_path="configs", config_name="exp/finetune")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    env = build_env(cfg.env, seed=cfg.seed)

    model = load_model(env, cfg.rl, load_path=cfg.model_path, seed=cfg.seed)

    model.tensorboard_log = "./tensorboard"

    model = train(model, cfg.rl)

    model.save("model")

    print("\nTraining finished!")
    print("\nRun: tensorboard --logdir outputs")

if __name__ == "__main__":
    main()
