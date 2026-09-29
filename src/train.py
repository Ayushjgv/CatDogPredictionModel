"""
GPU-accelerated training script with Automatic Mixed Precision (AMP),
checkpoint saving, and validation metrics.
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
)
from src.dataset import get_dataloaders
from src.model import get_model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training  ", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.3f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Evaluating", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train Cat vs Dog Model on GPU")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3", "custom_cnn"],
                        help="Model architecture (default: resnet18)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch without pretrained weights")
    args = parser.parse_args()

    device = DEVICE
    use_amp = not args.no_amp and device.type == "cuda"

    print("=" * 60)
    print(f"Starting Cat vs Dog Training on: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Mixed Precision (AMP) Enabled: {use_amp}")
    print(f"Architecture : {args.model}")
    print(f"Batch Size   : {args.batch_size}")
    print(f"Epochs       : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # 1. Dataloaders
    print("Loading datasets...")
    train_loader, test_loader, class_map = get_dataloaders(batch_size=args.batch_size)
    print(f"Classes: {class_map}")
    print(f"Training samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # 2. Model, Loss, Optimizer, Scaler
    model = get_model(args.model, num_classes=len(class_map), pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp=use_amp
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device, use_amp=use_amp
        )
        scheduler.step()

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}% | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_map": class_map,
            }, BEST_MODEL_PATH)
            print(f"  >>> Best model saved to {BEST_MODEL_PATH} (Accuracy: {val_acc * 100:.2f}%)")

    total_time = time.time() - start_total_time
    print("\n" + "=" * 60)
    print(f"Training Complete in {total_time / 60:.2f} minutes!")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Checkpoint location: {BEST_MODEL_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
