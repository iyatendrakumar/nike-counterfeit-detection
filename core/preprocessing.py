import cv2
import numpy as np

def preprocess_image(img):
    """
    Input: BGR image (OpenCV)
    Output: preprocessed grayscale image
    """
    if img is None:
        raise ValueError("Invalid image received")

    # Resize to fixed size
    img = cv2.resize(img, (300, 300))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Histogram equalization for lighting normalization
    gray = cv2.equalizeHist(gray)

    # Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray
