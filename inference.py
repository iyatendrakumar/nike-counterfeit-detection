import cv2
import numpy as np
import os

from core.preprocessing import preprocess_image
from core.roi import extract_roi
from core.features import extract_features
from core.classifier import load_model


def predict(image_path, domain="shoes"):
    """
    Predict authenticity of the product image.
    Returns: label, confidence, roi_image
    """

    model_path = f"models/{domain}_model.pkl"

    # ---- Safety check (deployment-safe) ----
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    model = load_model(model_path)

    # ---- Read image ----
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path or unreadable image.")

    # ---- Preprocess ----
    img = preprocess_image(img)
    roi = extract_roi(img, domain)
    features = extract_features(roi).reshape(1, -1)

    # ---- Prediction ----
    pred = model.predict(features)[0]
    decision_score = model.decision_function(features)[0]

    # ---- Distance-based confidence (stable) ----
    confidence = 1 / (1 + np.exp(-abs(decision_score)))
    confidence = round(confidence * 100, 2)

    label = "Genuine" if pred == 0 else "Counterfeit"

    return label, confidence, roi
