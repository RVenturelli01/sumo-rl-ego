import math
import random
import wandb


def log_histogram(data, title):
    table = wandb.Table(
        data=data,
        columns=["x", "y"],
    )
    histogram = wandb.plot.histogram(
        table,
        value="y",
        title=title,
    )

    wandb.log({title: histogram})
    

    
wandb.init()

data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]

log_histogram(data, "My Histogram")