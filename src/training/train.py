import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset.gtsrb_dataset import GTSRBDataset
from src.models.cnn_model import TrafficSignCNN
from src.models.resnet_model import create_resnet18
from src.utils.metrics import classification_metrics
from src.utils.transforms import get_eval_transforms, get_train_transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_classes: int, pretrained: bool):
    if model_name.lower() == "cnn":
        return TrafficSignCNN(num_classes=num_classes)
    if model_name.lower() == "resnet18":
        return create_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, metrics


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = classification_metrics(all_labels, all_preds)
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, metrics


def run_training(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Required fixed training settings.
    epochs = 20
    learning_rate = 0.001
    early_stopping_patience = int(config.get("early_stopping_patience", 5))

    set_seed(config.get("seed", 42))

    data_root = str(PROJECT_ROOT / config["paths"]["data_root"])
    image_size = int(config["image_size"])

    full_dataset = GTSRBDataset(
        data_root=data_root,
        csv_file=config["paths"]["train_csv"],
        transform=get_train_transforms(image_size=image_size),
        has_labels=True,
    )

    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    val_size = int(len(indices) * float(config["val_split"]))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(full_dataset, train_indices)

    val_base = GTSRBDataset(
        data_root=data_root,
        csv_file=config["paths"]["train_csv"],
        transform=get_eval_transforms(image_size=image_size),
        has_labels=True,
    )
    val_dataset = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=config["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    log_dir = PROJECT_ROOT / config["paths"]["tensorboard_log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    model_output = PROJECT_ROOT / "outputs" / "models" / "best_model.pth"
    model_output.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("f1/train", train_metrics["f1"], epoch)
        writer.add_scalar("f1/val", val_metrics["f1"], epoch)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": config["model_name"],
                    "num_classes": config["num_classes"],
                    "image_size": config["image_size"],
                },
                model_output,
            )
            print(f"Checkpoint updated: {model_output}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping triggered "
                f"after {epochs_without_improvement} epochs without validation improvement."
            )
            break

    writer.close()
    print(f"Training finished. Best model saved to: {model_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic sign classifier")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
        help="Path to training config yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.config)
