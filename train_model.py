import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Charger les images
img_dir = "dataset/dataset_FVC2000_DB4_B/dataset/train_data"
images = []
labels = []

for filename in os.listdir(img_dir):
    if filename.lower().endswith((".bmp", ".png", ".tif")):
        path = os.path.join(img_dir, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(filename.split("_")[0])

# HOG
hog_features = []
max_len = 0

for img in images:
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    max_len = max(max_len, len(features))
    hog_features.append(features)

# Padding
for i in range(len(hog_features)):
    if len(hog_features[i]) < max_len:
        hog_features[i] = np.pad(hog_features[i], (0, max_len - len(hog_features[i])))

X = np.array(hog_features)
le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = SVC(kernel='poly')
clf.fit(X_train, y_train)

# Sauvegarde
joblib.dump(clf, 'model.pkl')
print("Model trained and saved.")
