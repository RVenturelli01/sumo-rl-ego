
from stable_baselines3.common.callbacks import BaseCallback

class CustomLogsCallback(BaseCallback):

    def _on_step(self) -> bool:

        infos = self.locals.get("infos", [])
        metrics = {}

        for info in infos:
            if "log" in info:
                for k, v in info["log"].items():
                    metrics.setdefault(k, []).append(v)

        for k, values in metrics.items():
            self.logger.record(k, sum(values) / len(values))

        return True