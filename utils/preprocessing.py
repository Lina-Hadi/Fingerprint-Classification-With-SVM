import cv2
import numpy as np
from skimage.feature import hog
import joblib

def extract_hog_features(image_path):
    # Load HOG config from training
    hog_params = joblib.load("hog_params.pkl")
    max_len = joblib.load("max_len.pkl")
    
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Invalid image: {image_path}")
    
    # Match training preprocessing
    img = cv2.resize(img, (128, 128))
    features = hog(img, **hog_params)
    
    # Apply same padding as training
    if len(features) < max_len:
        features = np.pad(features, (0, max_len - len(features)))
    return features