import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_resnet18(num_classes: int = 43, pretrained: bool = True):
    """Create ResNet18 transfer learning model with replaced classification head."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
