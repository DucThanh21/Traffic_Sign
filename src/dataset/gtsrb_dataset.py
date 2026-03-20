from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GTSRBDataset(Dataset):
    """PyTorch Dataset for GTSRB training data (Train.csv)."""

    def __init__(
        self,
        data_root: str,
        csv_file: str = "Train.csv",
        transform: Optional[transforms.Compose] = None,
        has_labels: bool = True,
    ):
        self.data_root = Path(data_root)
        self.annotations = pd.read_csv(self.data_root / csv_file)
        self.has_labels = has_labels

        # Default preprocessing: resize -> tensor -> normalize.
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        if "Path" not in self.annotations.columns:
            raise ValueError("CSV must contain a 'Path' column.")
        if self.has_labels and "ClassId" not in self.annotations.columns:
            raise ValueError("CSV must contain a 'ClassId' column.")

    def __len__(self):
        return len(self.annotations)

    def _resolve_image_path(self, relative_path: str) -> Path:
        return self.data_root / relative_path

    def __getitem__(self, idx: int):
        row = self.annotations.iloc[idx]
        image_path = self._resolve_image_path(row["Path"])

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if not self.has_labels:
            return image, str(image_path)

        label = int(row["ClassId"])
        return image, label


def infer_num_classes(data_root: str, csv_file: str) -> int:
    annotations = pd.read_csv(Path(data_root) / csv_file)
    if "ClassId" not in annotations.columns:
        raise ValueError("Cannot infer classes because 'ClassId' column is missing.")
    return int(annotations["ClassId"].nunique())
