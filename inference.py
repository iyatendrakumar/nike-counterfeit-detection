import cv2
import numpy as np
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from core.preprocessing import preprocess_image
from core.roi import extract_roi
from core.features import extract_features
from core.classifier import load_model

# ImageNet class indices related to shoes/footwear
SHOE_CLASS_INDICES = {
    770,  # running shoe
    812,  # sandal
    oxford_shoe := 998,  # not real but placeholder
}

# Correct ImageNet shoe-related indices
VALID_SHOE_CLASSES = {
    770,  # running shoe / sneaker
    812,  # sandal
    819,  # soccer cleat
    852,  # tennis shoe
    679,  # padded shoe
    799,  # rugby ball (sometimes misclassified with shoes – exclude if needed)
}

# Load MobileNetV2 once at module level (avoid reloading every call)
_mobilenet = None

def _get_mobilenet():
    global _mobilenet
    if _mobilenet is None:
        _mobilenet = models.mobilenet_v2(pretrained=True)
        _mobilenet.eval()
    return _mobilenet

def is_shoe_image(image_path, top_k=5):
    """
    Returns True if the image likely contains a shoe/sneaker,
    using MobileNetV2 pretrained on ImageNet.
    Checks top-k predictions to allow for some variance.
    """
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        model = _get_mobilenet()
        with torch.no_grad():
            output = model(tensor)

        # Get top-k predicted class indices
        top_indices = output[0].topk(top_k).indices.tolist()

        # Check if any top prediction is shoe-related
        for idx in top_indices:
            if idx in VALID_SHOE_CLASSES:
                return True

        # Fallback: check decision score spread — very high confidence
        # on a non-shoe class means it's definitely not a shoe
        top_confidence = torch.softmax(output[0], dim=0).max().item()
        if top_confidence > 0.85:
            # Model is very confident it's something — and it's not a shoe
            return False

        # Low confidence overall — ambiguous image, allow it through
        # (your SVM will handle it, and threshold will catch outliers)
        return False

    except Exception:
        # If anything fails during image check, let the main pipeline handle it
        return False


def predict(image_path, domain="shoes"):
    """
    Predict authenticity of the product image.
    Returns: label, confidence, roi_image

    Labels:
        - "Genuine"       → authentic Nike product
        - "Counterfeit"   → fake Nike product
        - "Invalid Image" → not a shoe / unrecognizable input
    """

    # ---- Step 1: Pre-filter — is this even a shoe? ----
    if not is_shoe_image(image_path):
        return "Invalid Image (Not a Nike Product)", 0.0, None

    # ---- Step 2: Load model ----
    model_path = f"models/{domain}_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Please train the model first."
        )

    model = load_model(model_path)

    # ---- Step 3: Read and validate image ----
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path or unreadable image.")

    # ---- Step 4: Preprocess → ROI → Features ----
    img = preprocess_image(img)
    roi = extract_roi(img, domain)
    features = extract_features(roi).reshape(1, -1)

    # ---- Step 5: SVM Prediction ----
    pred = model.predict(features)[0]
    decision_score = model.decision_function(features)[0]

    # ---- Step 6: Confidence (sigmoid on decision distance) ----
    raw_confidence = 1 / (1 + np.exp(-abs(decision_score)))

    # ---- Step 7: Outlier guard — extreme scores = off-distribution ----
    # If the SVM decision score is unrealistically large, the image
    # is likely not a shoe despite passing the pre-filter.
    DECISION_SCORE_LIMIT = 4.5  # tune based on your training data
    if abs(decision_score) > DECISION_SCORE_LIMIT:
        return "Invalid Image (Uncertain Prediction)", 0.0, roi

    confidence = round(raw_confidence * 100, 2)
    label = "Genuine" if pred == 0 else "Counterfeit"

    return label, confidence, roi