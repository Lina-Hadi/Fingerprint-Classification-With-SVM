import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image_path, resize_dim=(128, 128), max_len=8100):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, resize_dim)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    if len(features) < max_len:
        features = np.pad(features, (0, max_len - len(features)))
    return features
