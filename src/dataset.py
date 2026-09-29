"""
Dataset loading and image preprocessing using torchvision ImageFolder.
Configured with GPU pin_memory for optimal CUDA throughput.
"""

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

def get_transforms():
    """Returns training and evaluation transformations."""
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    return train_transforms, val_transforms

def get_dataloaders(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
):
    """
    Initializes PyTorch DataLoaders for training and testing.
    Leaves the source dataset directory intact and loads images directly.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    train_tfms, test_tfms = get_transforms()

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_tfms)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader, train_dataset.class_to_idx

if __name__ == "__main__":
    print("Testing dataset loader...")
    try:
        train_loader, test_loader, class_map = get_dataloaders()
        print(f"Dataset classes: {class_map}")
        print(f"Training batches: {len(train_loader)} (~{len(train_loader.dataset)} images)")
        print(f"Test batches    : {len(test_loader)} (~{len(test_loader.dataset)} images)")
    except Exception as e:
        print(f"Dataset check note: {e}")
