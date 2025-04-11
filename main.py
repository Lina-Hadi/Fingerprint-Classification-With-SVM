import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Fingerprint Preprocessing", layout="centered")

st.title("🔍 Fingerprint Preprocessing")
st.write("Upload a fingerprint image to see the preprocessing steps.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_np = np.array(image)

    st.subheader("📷 Original Image")
    st.image(img_np, use_column_width=True, clamp=True)

    # Step 1: Gaussian Blur
    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)

    # Step 2: Otsu Thresholding
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    st.subheader("🧪 After Preprocessing")
    st.image(binary_img, use_column_width=True, caption="Gaussian Blur + Otsu Threshold", clamp=True)

    # Optional: Show histogram
    st.subheader("📊 Histogram of Original Image")
    hist_values, bins = np.histogram(img_np.flatten(), bins=256, range=[0,256])
    st.line_chart(hist_values)
else:
    st.info("👆 Please upload a fingerprint image to begin.")

# Later in prediction
proba = calibrated_svm.predict_proba(new_fingerprint)
max_confidence = max(proba[0])
if max_confidence < 0.7:
    st.warning("Fingerprint not recognized.")
else:
    predicted_class = calibrated_svm.predict(new_fingerprint)