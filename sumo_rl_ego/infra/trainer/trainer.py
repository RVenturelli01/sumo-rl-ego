from omegaconf import DictConfig, OmegaConf
from sumo_rl_ego.infra.utils.custom_logs_callback import CustomLogsCallback


def train(model, cfg: DictConfig):
    print("\n[INFRA] Starting training...")

    learn_kwargs = OmegaConf.to_container(cfg.training, resolve=True)

    model.learn(
        progress_bar=True,
        callback=CustomLogsCallback(),
        **learn_kwargs,
    )

    return model