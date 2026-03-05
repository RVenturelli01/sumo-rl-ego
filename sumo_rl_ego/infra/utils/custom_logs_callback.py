
from stable_baselines3.common.callbacks import BaseCallback


class CustomLogsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "log" in info:
                log_metrics = info["log"]

                for k, v in log_metrics.items():
                    self.logger.record(f"{k}", v)

        return True
    
