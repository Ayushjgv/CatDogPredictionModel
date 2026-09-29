"""
Inference and evaluation script using GPU acceleration.
Allows testing single images or benchmarking the test set.
"""

import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import BEST_MODEL_PATH, DEVICE
from src.dataset import get_transforms, get_dataloaders
from src.model import get_model

def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Loads model checkpoint and restores architecture."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Train a model first!")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    class_map = checkpoint.get("class_map", {"cats": 0, "dogs": 1})
    idx_to_class = {v: k for k, v in class_map.items()}

    model = get_model(model_name, num_classes=len(class_map), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class, checkpoint.get("val_acc", 0.0)

def predict_single_image(image_path: Path, model, idx_to_class, device):
    """Runs GPU inference on a single image file."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    _, val_tfms = get_transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = val_tfms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]
        pred_idx = probabilities.argmax().item()
        confidence = probabilities[pred_idx].item()

    pred_label = idx_to_class[pred_idx]
    return pred_label, confidence, probabilities

def evaluate_test_set(model, device):
    """Runs full test set evaluation and reports top metrics."""
    print("Loading test dataset...")
    _, test_loader, class_map = get_dataloaders()
    idx_to_class = {v: k for k, v in class_map.items()}

    correct = 0
    total = 0
    print(f"Evaluating {len(test_loader.dataset)} test images on {device}...")

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print("=" * 50)
    print(f"Test Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Cat vs Dog GPU Inference")
    parser.add_argument("--image", type=str, help="Path to single image for prediction")
    parser.add_argument("--eval-test", action="store_true", help="Run full evaluation on test dataset")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_MODEL_PATH), help="Path to model weights")
    args = parser.parse_args()

    device = DEVICE
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[!] Checkpoint not found: {ckpt_path}")
        print("Please run training first using: python -m src.train")
        return

    print(f"Loading checkpoint from: {ckpt_path} onto {device}...")
    model, idx_to_class, val_acc = load_checkpoint(ckpt_path, device)
    print(f"Model loaded successfully! (Trained accuracy: {val_acc * 100:.2f}%)")

    if args.image:
        img_path = Path(args.image)
        label, conf, probs = predict_single_image(img_path, model, idx_to_class, device)
        print("=" * 50)
        print(f"Prediction for '{img_path.name}':")
        print(f"  Class      : {label.upper()}")
        print(f"  Confidence : {conf * 100:.2f}%")
        print("  Probabilities:")
        for idx, score in enumerate(probs):
            print(f"    - {idx_to_class[idx]}: {score.item() * 100:.2f}%")
        print("=" * 50)

    elif args.eval_test:
        evaluate_test_set(model, device)
    else:
        print("Please provide --image <path> or --eval-test. Run with --help for options.")

if __name__ == "__main__":
    main()
