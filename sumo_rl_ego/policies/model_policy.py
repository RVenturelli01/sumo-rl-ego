from .base_policy import Policy


class ModelPolicy(Policy):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action