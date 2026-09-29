"""
Flask backend for the Cat vs Dog classifier web app.

Routes:
    GET  /            -> the upload page
    GET  /api/health  -> readiness probe with model metadata
    POST /api/predict -> accepts an image upload and returns the prediction

Run with:
    python app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (  # noqa: E402
    CLASS_NAMES,
    IMAGE_SIZE_LIMIT_MB,
    CatDogPredictor,
    InvalidImageError,
    load_image,
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp", "bmp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("catdog-web")


def create_app(predictor: CatDogPredictor | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = IMAGE_SIZE_LIMIT_MB * 1024 * 1024

    if predictor is None:
        logger.info("Loading model weights...")
        predictor = CatDogPredictor()
    app.config["PREDICTOR"] = predictor
    logger.info(f"Model ready on {predictor.device_name}. Classes: {', '.join(CLASS_NAMES)}")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "device": predictor.device_name,
                "classes": CLASS_NAMES,
            }
        )

    @app.post("/api/predict")
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No image was uploaded."}), 400

        upload = request.files["file"]
        if not upload.filename:
            return jsonify({"error": "No image was selected."}), 400

        extension = upload.filename.rsplit(".", 1)[-1].lower() if "." in upload.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return jsonify({"error": f"Unsupported file type. Allowed: {allowed}."}), 400

        try:
            image = load_image(upload.read())
        except InvalidImageError as exc:
            return jsonify({"error": str(exc)}), 400

        try:
            result = predictor.predict(image)
        except Exception:
            logger.exception("Prediction failed")
            return jsonify({"error": "Prediction failed on the server."}), 500

        logger.info(f"Predicted '{result['label']}' at {result['confidence'] * 100:.1f}%")
        return jsonify({**result, "filename": upload.filename})

    @app.errorhandler(413)
    def too_large(_error):
        limit = IMAGE_SIZE_LIMIT_MB
        return jsonify({"error": f"Image is too large. Maximum size is {limit} MB."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    port = 5000
    app.run(host="127.0.0.1", port=port, debug=False)
