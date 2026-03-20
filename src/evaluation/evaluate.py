import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset.gtsrb_dataset import GTSRBDataset
from src.models.cnn_model import TrafficSignCNN
from src.models.resnet_model import create_resnet18
from src.utils.metrics import classification_metrics
from src.utils.transforms import get_eval_transforms


def load_model(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    num_classes = checkpoint.get("num_classes", 43)

    if model_name == "cnn":
        model = TrafficSignCNN(num_classes=num_classes)
    else:
        model = create_resnet18(num_classes=num_classes, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def run_evaluation(config_path: str, checkpoint_path: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = PROJECT_ROOT / config["paths"]["data_root"]
    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path
        else PROJECT_ROOT / config["paths"]["model_output"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(ckpt, device)

    image_size = checkpoint.get("image_size", config["image_size"])

    test_dataset = GTSRBDataset(
        data_root=str(data_root),
        csv_file=config["paths"]["test_csv"],
        transform=get_eval_transforms(image_size=image_size),
        has_labels=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    metrics = classification_metrics(y_true, y_pred)
    print("Evaluation metrics on test set:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.config, args.checkpoint)
