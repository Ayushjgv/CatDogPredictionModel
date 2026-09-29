"""
Neural network model architectures for Cat vs Dog classification.
Supports Transfer Learning (ResNet18, MobileNetV3) and a Custom CNN.
"""

import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    """Lightweight custom convolutional network for training from scratch."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 256x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name: str = "resnet18", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function to instantiate models.
    Options: 'resnet18' (recommended), 'mobilenet_v3', 'custom_cnn'.
    """
    model_name = model_name.lower()
    
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "custom_cnn":
        return SimpleCNN(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from 'resnet18', 'mobilenet_v3', 'custom_cnn'.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Instantiating ResNet18 model on {device}...")
    model = get_model("resnet18", num_classes=2, pretrained=False).to(device)
    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    out = model(dummy_input)
    print(f"Forward pass output shape: {out.shape} -> Expected [2, 2]")
