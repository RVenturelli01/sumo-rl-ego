from modules.policies.base import BasePolicy


class SB3Policy(BasePolicy):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
