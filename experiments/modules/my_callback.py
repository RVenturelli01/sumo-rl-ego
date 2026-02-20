
from stable_baselines3.common.callbacks import BaseCallback

class MyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "global_metrics" in info:
                metrics = info["global_metrics"]

                for k, v in metrics.items():
                    self.logger.record(f"metrics/{k}", v)

        return True
    
