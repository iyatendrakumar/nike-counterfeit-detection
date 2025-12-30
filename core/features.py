import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ORB detector
orb = cv2.ORB_create(nfeatures=500)

def extract_features(img):
    """
    Returns a fixed-length feature vector
    """

    features = []

    # ---------- ORB FEATURES ----------
    keypoints, descriptors = orb.detectAndCompute(img, None)
    num_keypoints = len(keypoints) if keypoints else 0
    features.append(num_keypoints)

    if descriptors is not None:
        features.append(descriptors.mean())
        features.append(descriptors.std())
    else:
        features.extend([0, 0])

    # ---------- EDGE DENSITY ----------
    edges = cv2.Canny(img, 100, 200)
    edge_density = edges.sum() / edges.size
    features.append(edge_density)

    # ---------- GRAYSCALE HISTOGRAM ----------
    hist = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist = hist.flatten()
    hist = hist / (hist.sum() + 1e-6)
    features.extend(hist.tolist())

    # ---------- TEXTURE (LBP) ----------
    lbp = local_binary_pattern(img, P=8, R=1, method="uniform")
    features.append(lbp.var())

    return np.array(features, dtype=np.float32)
