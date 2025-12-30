import os
import cv2
import numpy as np

from core.preprocessing import preprocess_image
from core.roi import extract_roi
from core.features import extract_features
from core.classifier import train_model

def train_domain(domain):
    X, y = [], []
    base_path = f"dataset/{domain}"

    for label, cls in enumerate(["genuine", "fake"]):
        folder = os.path.join(base_path, cls)

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = preprocess_image(img)
            roi = extract_roi(img, domain)
            features = extract_features(roi)

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    model_path = f"models/{domain}_model.pkl"
    train_model(X, y, model_path)

    print(f"✅ {domain.upper()} model trained | Samples: {len(X)}")

if __name__ == "__main__":
    train_domain("shoes")
   
