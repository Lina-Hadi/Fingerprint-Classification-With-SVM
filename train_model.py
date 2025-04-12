import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load images with validation
img_dir = "dataset/dataset_FVC2000_DB4_B/dataset/train_data"
images = []
labels = []

for filename in os.listdir(img_dir):
    if filename.lower().endswith((".bmp", ".png", ".tif")):
        path = os.path.join(img_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipped corrupt file: {filename}")
            continue
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(filename.split("_")[0])

# Validate dataset
if not images:
    raise ValueError("No valid images found in dataset directory")

# HOG feature extraction
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

hog_features = [hog(img, **hog_params) for img in images]
max_len = max(len(f) for f in hog_features)

# Apply padding
X = np.array([np.pad(f, (0, max_len - len(f))) for f in hog_features])
le = LabelEncoder()
y = le.fit_transform(labels)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Use calibrated classifier
base_clf = LinearSVC(class_weight='balanced', C=0.1, max_iter=10000)
clf = CalibratedClassifierCV(base_clf, method='sigmoid')
clf.fit(X_train, y_train)

# Save all components
joblib.dump(clf, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(max_len, "max_len.pkl")
joblib.dump(hog_params, "hog_params.pkl")
print("✅ Correctly saved trained components")