"""
Main training script.
"""

import torch
from torch import nn, optim
from torchvision import transforms
import wandb

from data import get_dataloaders
from modules import ResNet


def make(data_dir, config, device):
    """Create model, dataloaders, loss function, optimizer, and scheduler."""

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )

    pin_memory = device == "cuda:0"
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_dir, train_transform, test_transform, config.batch_size, True, pin_memory
    )

    model = ResNet(config.n)
    model.to(device)

    loss_func = nn.NLLLoss2d()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=config.gamma
    )

    return (
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        loss_func,
        optimizer,
        scheduler,
    )


def evaluate(model, loader, loss_func, device):
    """Evaluates the model's loss and error on a dataset"""

    model.eval()
    total_loss = 0.0
    total_wrong = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            total_loss += loss_func(outputs, labels) * labels.size(0)

            _, pred = torch.max(outputs.data, 1)
            total_wrong += (pred != labels).sum().item()

    loss = total_loss / len(loader.dataset)
    error = total_wrong / len(loader.dataset)

    return loss, error


def train(
    model, train_loader, val_loader, loss_func, optimizer, scheduler, config, device
):
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
        train_loss, train_error = evaluate(model, train_loader, loss_func, device)
        val_loss, val_error = evaluate(model, val_loader, loss_func, device)
        wandb.log(
            {
                "epoch": epoch,
                "train/error": train_error,
                "train/loss": train_loss,
                "validation/error": val_error,
                "validation/loss": val_loss,
            }
        )

        # Adjust learning rate
        scheduler.step()


def main():
    """Starts a training run"""

    # Ensure deterministic behavior
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.login()

    project = "resnet"
    data_dir = "./data"
    model_name = "resnet"
    model_path = "model.pth"
    config = {
        "n": 3,
        "batch_size": 128,
        "lr": 0.1,
        "epochs": 182,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "milestones": [91, 137],
        "gamma": 0.1,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    with wandb.init(project=project, config=dict(config)) as run:
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
        ) = make(data_dir, config, device)

        # and use them to train the model
        train(
            model,
            train_loader,
            val_loader,
            loss_func,
            optimizer,
            scheduler,
            config,
            device,
        )

        # Evaluate the model on the test set
        _, test_error = evaluate(model, test_loader, loss_func, device)
        wandb.log({"test/error": test_error})

        # Save model weights
        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            metadata=dict(config),
        )

        torch.save(model.state_dict(), model_path)
        model_artifact.add_file(model_path)
        wandb.save(model_path)
        run.log_artifact(model_artifact)


if __name__ == "__main__":
    main()
