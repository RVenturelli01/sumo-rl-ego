from sumo_rl_ego.infra.utils.custom_logs_callback import CustomLogsCallback


def train(model, config_rl):

    learn_kwargs = config_rl.get("training", {}).copy()

    model.learn(
        progress_bar=True,
        callback=CustomLogsCallback(),
        **learn_kwargs,
    )

    return model