
import random
import wandb

# Generate random data for the table
data = [
    ["car", random.uniform(0, 1)],
    ["bus", random.uniform(0, 1)],
    ["road", random.uniform(0, 1)],
    ["person", random.uniform(0, 1)],
]

# Create a table with the data
table = wandb.Table(data=data, columns=["class", "accuracy"])

# Initialize a W&B run and log the bar plot
with wandb.init(project="bar_chart") as run:
    # Create a bar plot from the table
    bar_plot = wandb.plot.bar(
        table=table,
        label="class",
        value="accuracy",
        title="Object Classification Accuracy",
    )

    # Log the bar chart to W&B
    run.log({"bar_plot": bar_plot})