"""
Central configuration for dataset paths, model hyperparameters, and GPU device setup.
"""

from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths (referencing the existing dataset folder without modifying it)
DATA_DIR = BASE_DIR / "dog-cat-full-dataset-master" / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Directory to save trained model weights
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# Image preprocessing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Training hyperparameters (tuned for RTX 3050 6GB)
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# Hardware device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()

# Classes
CLASS_NAMES = ["cat", "dog"]
