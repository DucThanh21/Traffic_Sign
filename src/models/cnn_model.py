import torch.nn as nn


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Input image is 64x64 -> after two pool(2) layers it becomes 16x16.
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.fc(x)
