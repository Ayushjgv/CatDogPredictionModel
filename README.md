# Cat vs Dog Deep Learning (GPU Accelerated)

This project trains and evaluates a deep convolutional neural network for binary image classification (Cat vs Dog) with full hardware acceleration on NVIDIA GPUs using PyTorch and CUDA.

---

## 1. Quickstart & Activation

The Python 3.12 virtual environment is located in `.venv` in this directory.

### In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### In Command Prompt (cmd):
```cmd
.\.venv\Scripts\activate.bat
```

---

## 2. Verify GPU & CUDA Acceleration

To confirm that PyTorch detects your NVIDIA GeForce RTX 3050 GPU and performs matrix multiplication on CUDA:

```powershell
.\.venv\Scripts\python check_gpu.py
```

Expected output:
- `CUDA Available: True`
- `GPU Device [0]: NVIDIA GeForce RTX 3050 ...`
- `Tensor computation successful ...`

---

## 3. Project Structure

```
d:\AIML\CatDogPrediction\
├── .venv/                         # Isolated Python 3.12 virtual environment
├── requirements.txt               # Dependencies with PyTorch CUDA index
├── check_gpu.py                   # Hardware diagnostic script
├── dog-cat-full-dataset-master/   # Unaltered dataset folder
│   └── data/
│       ├── train/ (cats, dogs)
│       └── test/  (cats, dogs)
├── src/
│   ├── config.py                  # Hyperparameters and hardware config
│   ├── dataset.py                 # PyTorch DataLoader with augmentations & pin_memory
│   ├── model.py                   # Transfer learning (ResNet18, MobileNetV3) & Custom CNN
│   ├── train.py                   # GPU training pipeline with Mixed Precision (AMP)
│   └── predict.py                 # Single-image prediction & test-set evaluation
├── checkpoints/                   # Saved best model weights (.pth)
└── README.md
```

---

## 4. Training the Model on GPU

Run training with default settings (ResNet-18 with pretrained weights, Mixed Precision AMP enabled):

```powershell
.\.venv\Scripts\python -m src.train
```

### Custom Training Arguments:
- **Change Model:** `.\.venv\Scripts\python -m src.train --model mobilenet_v3` or `--model custom_cnn`
- **Set Epochs & Batch Size:** `.\.venv\Scripts\python -m src.train --epochs 5 --batch-size 32`
- **Adjust Learning Rate:** `.\.venv\Scripts\python -m src.train --lr 0.0001`
- **Train from scratch (no pretrained weights):** `.\.venv\Scripts\python -m src.train --no-pretrained`

The best performing model weights will automatically be saved to `checkpoints/best_model.pth`.

---

## 5. Testing & Inference

### Predict a Single Image:
```powershell
.\.venv\Scripts\python -m src.predict --image "dog-cat-full-dataset-master/data/test/cats/cat.1500.jpg"
```

### Evaluate the Entire Test Set:
```powershell
.\.venv\Scripts\python -m src.predict --eval-test
```
