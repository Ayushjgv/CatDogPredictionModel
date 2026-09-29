# Cat vs Dog Classifier — Web App

A Flask web interface that serves the trained `CatDogCNN` model in
`cat_dog_model.pth`. Upload a photo, get back **cat** or **dog** with a
confidence score and per-class probability bars.

## How it works

| Layer | Shape |
| --- | --- |
| Conv(3→32) + ReLU + MaxPool | 128×128 → 64×64 |
| Conv(32→64) + ReLU + MaxPool | 64×64 → 32×32 |
| Conv(64→128) + ReLU + MaxPool | 32×32 → 16×16 |
| Conv(128→256) + ReLU + MaxPool | 16×16 → 8×8 |
| Flatten + Linear(16384→512) + ReLU + Dropout(0.5) + Linear(512→2) | 2 logits |

Preprocessing matches training exactly: `Resize((128,128))` → `ToTensor()` →
`Normalize(mean=0.5, std=0.5)`. Class index 0 is `cat`, 1 is `dog`
(alphabetical order of the `cats` / `dogs` folders).

Measured accuracy on the test set is ~91%.

## Setup

The project virtual environment already has PyTorch installed; only Flask is
missing.

```bash
..\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000>.

On Windows you can double-click `start.bat` instead.

## Files

```
website/
├── app.py             Flask routes: /, /api/health, /api/predict
├── model.py           CatDogCNN architecture, checkpoint loading, inference
├── cat_dog_model.pth  Trained weights (raw state_dict)
├── templates/
│   └── index.html     Page markup
├── static/
│   ├── style.css      Styling (dark theme, glassmorphism)
│   └── script.js      Drag-and-drop upload, preview, result rendering
├── requirements.txt
└── start.bat
```

## API

**`POST /api/predict`** — `multipart/form-data` with a `file` field.

```json
{
  "label": "dog",
  "confidence": 0.9871,
  "probabilities": { "cat": 0.0129, "dog": 0.9871 },
  "filename": "rex.jpg"
}
```

**`GET /api/health`**

```json
{ "status": "ok", "device": "CUDA GPU", "classes": ["cat", "dog"] }
```

Errors return `{"error": "..."}` with status 400 (bad upload), 413 (over
8 MB), or 500.
