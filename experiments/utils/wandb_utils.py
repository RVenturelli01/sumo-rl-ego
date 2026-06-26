import wandb
import numpy as np


def log_histogram(data, value, title):
    data=[[i, float(v)] for i, v in enumerate(data)]

    table = wandb.Table(
        data=data,
        columns=["x", value],
    )
    histogram = wandb.plot.histogram(
        table,
        value=value,
        title=title,
    )

    print(f"Logging histogram: {title}")
    wandb.log({title: histogram})
    

def log_bar_plot(data, value, title):
    
    table = wandb.Table(
        data=data,
        columns=["class", value],
    )

    bar_plot = wandb.plot.bar(
        table,
        label="class",        
        value=value,   
        title=title,
    )

    print(f"Logging bar plot: {title}")
    wandb.log({title: bar_plot})
