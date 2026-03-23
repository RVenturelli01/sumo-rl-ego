from .base_policy import Policy


class ModelPolicy(Policy):
    def __init__(self, model_or_callable, deterministic=True):
        super().__init__()
        self.model_or_callable = model_or_callable
        self.deterministic = deterministic

    def reset(self):
        reset_fn = getattr(self.model_or_callable, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def predict(self, obs):
        if hasattr(self.model_or_callable, "predict"):
            predict_fn = self.model_or_callable.predict

            try:
                output = predict_fn(obs, deterministic=self.deterministic)
            except TypeError:
                output = predict_fn(obs)
        else:
            output = self.model_or_callable(obs)

        if isinstance(output, tuple):
            action = output[0]
        else:
            action = output

        return action
