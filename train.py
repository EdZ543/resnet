import torch
from train_utils import model_pipeline
import wandb

wandb.login()

config = dict(
    {
        "n": 3,
        "batch_size": 128,
        "learning_rate": 0.1,
        "epochs": 182,
        "weight_decay": 0.0001,
        "momentum": 0.9,
    },
)

model_pipeline("ResNet-20", config)