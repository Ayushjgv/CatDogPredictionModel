"""
Model loading and inference for the Cat vs Dog classifier.

The weights in cat_dog_model.pth were trained with the CatDogCNN architecture
defined in code.ipynb: four Conv/ReLU/MaxPool blocks down to an 8x8x256 feature
map, flattened into a 512-unit head, then a 2-class output layer.

Input contract (must match training exactly):
    - RGB
    - resized to 128x128
    - scaled to [0, 1] by ToTensor
    - normalized with mean=0.5, std=0.5  ->  roughly [-1, 1]

Class indices come from ImageFolder over the `cats` / `dogs` folders, which
sorts alphabetically, so index 0 is "cat" and index 1 is "dog".
"""

from __future__ import annotations

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

MODEL_PATH = Path(__file__).resolve().parent / "cat_dog_model.pth"

IMAGE_SIZE = (128, 128)
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

CLASS_NAMES = ["cat", "dog"]
IMAGE_SIZE_LIMIT_MB = 8


class CatDogCNN(nn.Module):
    """Convolutional network used to train cat_dog_model.pth."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_transform() -> transforms.Compose:
    """Preprocessing pipeline; must stay in sync with the training transforms."""
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


class CatDogPredictor:
    """Loads the checkpoint once and serves predictions from any thread."""

    def __init__(self, model_path: Path = MODEL_PATH, device: torch.device | None = None):
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = build_transform()

        self.model = CatDogCNN(num_classes=len(CLASS_NAMES))
        # weights_only=True is safe here: the file is a plain tensor state_dict.
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @property
    def device_name(self) -> str:
        return "CUDA GPU" if self.device.type == "cuda" else "CPU"

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        """Classify a PIL image and return the label, confidence, and per-class scores."""
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())

        return {
            "label": CLASS_NAMES[index],
            "confidence": float(probabilities[index].item()),
            "probabilities": {
                name: float(probabilities[i].item()) for i, name in enumerate(CLASS_NAMES)
            },
        }


class InvalidImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an image."""


def load_image(source: bytes | io.BytesIO) -> Image.Image:
    """Decode uploaded bytes into a PIL image, guarding against malformed files."""
    try:
        image = Image.open(io.BytesIO(source))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("That file could not be read as an image.") from exc

    if image.width * image.height > 40_000_000:
        raise InvalidImageError("Image resolution is too large to process.")

    # Phone cameras store portrait shots as landscape pixels plus an EXIF
    # "rotate me" tag. PIL does not apply it, so a sideways animal would reach
    # the model unless we transpose it here.
    return ImageOps.exif_transpose(image).convert("RGB")
