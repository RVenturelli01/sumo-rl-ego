from stable_baselines3 import DQN, PPO
from experiments.policies.base_policy import BasePolicy

ALGOS = {
    "DQN": DQN,
    "PPO": PPO,
}

class SB3Policy(BasePolicy):
    def __init__(self, model_path: str, algo: str = "DQN"):
        super().__init__()
        self.model = ALGOS[algo.upper()].load(model_path)

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
