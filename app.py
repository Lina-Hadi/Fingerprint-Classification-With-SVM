import os
import cv2
import numpy as np
import joblib
import streamlit as st
from skimage.feature import hog

# Load the trained model
model = joblib.load('./model.pkl')

# Define a function to handle predictions
def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize to match the input size for the model
    
    # Extract HOG features from the uploaded image
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
    
    # Predict the label
    prediction = model.predict([features])
    return prediction[0]

# Streamlit interface
def main():
    # Set the title of the app
    st.title("Fingerprint Classification with SVM")

    # Display a file uploader widget for the user to upload an image
    uploaded_file = st.file_uploader("Choose a fingerprint image", type=["bmp", "png", "tif"])
    
    if uploaded_file is not None:
        # Read and display the image
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image.", use_column_width=True)
        
        # Save the uploaded image temporarily
        temp_img_path = "temp_image"
        with open(temp_img_path, 'wb') as f:
            f.write(image_bytes)

        # Predict the label
        label = predict_image(temp_img_path)
        st.write(f"Predicted label: {label}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
