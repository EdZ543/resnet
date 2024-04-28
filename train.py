"""
Main training script.
"""

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import wandb

from dataloader import get_dataloader
from resnet import ResNet

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make(config):
    """Create model, dataloaders, loss function, optimizer, and scheduler."""

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # Paper uses per-pixel mean subtraction
            # I use PyTorch's normalization
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(32, padding=4),
        ]
    )
    train_dataloader, test_dataloader = get_dataloader(
        train_transform, transforms.ToTensor(), config.batch_size
    )

    model = ResNet(config.n).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    scheduler = MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )

    return (
        model,
        train_dataloader,
        test_dataloader,
        loss_func,
        optimizer,
        scheduler,
    )


def evaluate(model, loader, loss_func):
    """Evaluates the model's loss and error on a dataset"""

    model.eval()
    loss_sum = 0.0
    with torch.inference_mode():
        wrong = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_sum += loss_func(outputs, labels) * labels.size(0)

            _, predicted_indices = torch.max(outputs.data, 1)
            wrong += (predicted_indices != labels).sum().item()

    loss = loss_sum / len(loader.dataset)
    error = wrong / len(loader.dataset)

    return loss, error


def train(model, train_loader, test_loader, loss_func, optimizer, scheduler, config):
    """Trains the model for the specified number of epochs."""

    for epoch in range(config.epochs):
        model.train()

        # Train for one epoch
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # Log training and testing metrics
        train_loss, train_error = evaluate(model, train_loader, loss_func)
        test_loss, test_error = evaluate(model, test_loader, loss_func)
        wandb.log(
            {
                "epoch": epoch,
                "train/error": train_error,
                "train/loss": train_loss,
                "test/error": test_error,
                "test/loss": test_loss,
            }
        )

        # Adjust learning rate
        scheduler.step()


def model_pipeline(project, model_name, config):
    """Trains a model and logs artifacts and metrics."""

    with wandb.init(project=project, config=dict(config)) as run:
        config = wandb.config

        # make the model, data, optimizer, and scheduler
        (
            model,
            train_loader,
            test_loader,
            loss_func,
            optimizer,
            scheduler,
        ) = make(config)

        # and use them to train the model
        train(model, train_loader, test_loader, loss_func, optimizer, scheduler, config)

        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            metadata=dict(config),
        )

        # Save model weights
        torch.save(model.state_dict(), model_name + ".pth")
        model_artifact.add_file(model_name + ".pth")
        wandb.save(model_name + ".pth")
        run.log_artifact(model_artifact)


# Execute training pipeline
if __name__ == "__main__":
    wandb.login()

    hyperparameters = {
        "n": 3,
        "batch_size": 128,
        "learning_rate": 0.1,
        "epochs": 164,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "lr_milestones": [82, 123],
        "lr_gamma": 0.1,
    }

    model_pipeline("resnet", "resnet", hyperparameters)
