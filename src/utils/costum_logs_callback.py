
from stable_baselines3.common.callbacks import BaseCallback


class CostumLogsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "rollout_metrics" in info:
                metrics = info["rollout_metrics"]

                for k, v in metrics.items():
                    self.logger.record(f"{k}", v)

        return True
    
