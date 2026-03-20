import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.cnn_model import TrafficSignCNN
from src.models.resnet_model import create_resnet18
from src.utils.transforms import get_eval_transforms


# Standard GTSRB class names (43 classes).
GTSRB_CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

GTSRB_CLASS_DESCRIPTIONS_VI = {
    0: "Gioi han toc do toi da 20 km/h",
    1: "Gioi han toc do toi da 30 km/h",
    2: "Gioi han toc do toi da 50 km/h",
    3: "Gioi han toc do toi da 60 km/h",
    4: "Gioi han toc do toi da 70 km/h",
    5: "Gioi han toc do toi da 80 km/h",
    6: "Het han gioi han toc do 80 km/h",
    7: "Gioi han toc do toi da 100 km/h",
    8: "Gioi han toc do toi da 120 km/h",
    9: "Cam vuot",
    10: "Cam xe tren 3.5 tan vuot",
    11: "Duong uu tien tai giao lo tiep theo",
    12: "Duong uu tien",
    13: "Nhuong duong",
    14: "Dung lai",
    15: "Cam tat ca phuong tien",
    16: "Cam xe tren 3.5 tan",
    17: "Cam di vao",
    18: "Canh bao nguy hiem chung",
    19: "Canh bao cua vong ben trai",
    20: "Canh bao cua vong ben phai",
    21: "Canh bao nhieu cua vong",
    22: "Duong gom ghe",
    23: "Duong tron truot",
    24: "Duong hep ben phai",
    25: "Cong truong",
    26: "Den giao thong",
    27: "Nguoi di bo",
    28: "Tre em qua duong",
    29: "Xe dap qua duong",
    30: "Canh bao bang/tuyet",
    31: "Canh bao dong vat hoang da",
    32: "Het tat ca gioi han toc do va cam vuot",
    33: "Huong phai phia truoc",
    34: "Huong trai phia truoc",
    35: "Chi duoc di thang",
    36: "Di thang hoac re phai",
    37: "Di thang hoac re trai",
    38: "Di ben phai",
    39: "Di ben trai",
    40: "Vong xuyen bat buoc",
    41: "Het cam vuot",
    42: "Het cam xe tren 3.5 tan vuot",
}


def load_class_label_map(data_root: Path) -> dict[int, str]:
    """Build ClassId -> label mapping from data/Meta.csv when available."""
    meta_csv = data_root / "Meta.csv"
    if not meta_csv.exists():
        return {}

    meta_df = pd.read_csv(meta_csv)
    if "ClassId" not in meta_df.columns:
        return {}

    label_map = {}
    for _, row in meta_df.iterrows():
        class_id = int(row["ClassId"])
        label_map[class_id] = GTSRB_CLASS_NAMES.get(class_id, f"class_{class_id}")

    return label_map


def get_class_metadata(class_id: int) -> dict[str, str | int]:
    return {
        "class_id": class_id,
        "name_en": GTSRB_CLASS_NAMES.get(class_id, f"class_{class_id}"),
        "description_vi": GTSRB_CLASS_DESCRIPTIONS_VI.get(class_id, "Khong co mo ta"),
    }


def load_checkpoint_model(checkpoint_path: Path, device):
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


def predict_image(model, image_path: Path, image_size: int, device):
    transform = get_eval_transforms(image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

    return pred_idx, confidence


def resolve_class_label(pred_idx: int, class_label_map: dict[int, str]) -> str:
    return class_label_map.get(pred_idx, GTSRB_CLASS_NAMES.get(pred_idx, f"class_{pred_idx}"))


def run_prediction(config_path: str, image_path: str, checkpoint_path: str | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path
        else PROJECT_ROOT / config["paths"]["model_output"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(ckpt, device)
    image_size = checkpoint.get("image_size", config["image_size"])
    data_root = PROJECT_ROOT / config["paths"]["data_root"]
    class_label_map = load_class_label_map(data_root)

    pred_idx, confidence = predict_image(model, Path(image_path), image_size, device)
    class_label = resolve_class_label(pred_idx, class_label_map)

    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class label: {class_label}")
    print(f"Confidence: {confidence:.4f}")
    return class_label


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for a single image")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "train_config.yaml"),
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(args.config, args.image, args.checkpoint)
