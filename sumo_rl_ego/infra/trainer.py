from omegaconf import DictConfig, OmegaConf
from .custom_logs_callback import CustomLogsCallback


def train(model, cfg: DictConfig):
    print("[SRE] Starting training...")

    learn_kwargs = OmegaConf.to_container(cfg.training, resolve=True)

    model.learn(
        progress_bar=True,
        callback=CustomLogsCallback(),
        **learn_kwargs,
    )

    return model