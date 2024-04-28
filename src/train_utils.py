import random
import math

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import wandb

from dataloader import get_dataloader
from resnet import ResNet

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make(config):
    train_dataloader, val_dataloader = get_dataloader(True, config.batch_size)
    test_dataloader = get_dataloader(False, config.batch_size)

    model = ResNet(config.n).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    scheduler = MultiStepLR(optimizer, milestones=[91, 137], gamma=0.1)

    return (
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_func,
        optimizer,
        scheduler,
    )


def evaluate(model, loader, loss_func):
    model.eval()
    loss_sum = 0.0
    with torch.inference_mode():
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_sum += loss_func(outputs, labels) * labels.size(0)

            _, predicted_indices = torch.max(outputs.data, 1)
            correct += (predicted_indices == labels).sum().item()

    loss = loss_sum / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    return loss, accuracy


def train(model, train_loader, val_loader, loss_func, optimizer, scheduler, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_func, log="all", log_freq=10)

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)
    for epoch in range(config.epochs):
        model.train()

        # Train for one epoch
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Report metrics every 25th batch
            if ((step + 1) % 25) == 0:
                accuracy = (outputs.argmax(1) == labels).float().mean()
                wandb.log(
                    {
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch))
                        / n_steps_per_epoch,
                        "train/error": 1 - accuracy,
                        "train/train_loss": train_loss,
                    }
                )

        # Validate
        val_loss, val_accuracy = evaluate(model, val_loader, loss_func)
        wandb.log(
            {
                "validation/error": 1 - val_accuracy,
                "validation/train_loss": val_loss,
            }
        )

        # Adjust learning rate
        scheduler.step()


def model_pipeline(entity, project, job_name, config):
    settings = wandb.Settings(job_name=job_name)

    # tell wandb to get started
    with wandb.init(entity=entity, project=project, settings=settings, config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, optimizer, and scheduler
        (
            model,
            train_loader,
            val_loader,
            test_loader,
            loss_func,
            optimizer,
            scheduler,
        ) = make(config)

        # and use them to train the model
        train(model, train_loader, val_loader, loss_func, optimizer, scheduler, config)

        # and test its final performance
        # Validate
        _, test_accuracy = evaluate(model, test_loader, loss_func)
        wandb.log(
            {
                "test/error": 1 - test_accuracy,
            }
        )

        wandb.run.log_code()

    return model
