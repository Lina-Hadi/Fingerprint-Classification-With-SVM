import os
import joblib
import streamlit as st
import numpy as np


# Load components
model = joblib.load('./model.pkl')
le = joblib.load('./label_encoder.pkl')
from utils.preprocessing import extract_hog_features

def main():
    st.title("Fingerprint Authentication System")
    
    uploaded_file = st.file_uploader("Upload fingerprint", type=["bmp", "png", "tif"])
    
    if uploaded_file:
        # Save to temporary file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Feature extraction
            features = extract_hog_features(temp_path)
            
            # Get probability scores
            proba = model.predict_proba([features])[0]
            confidence = np.max(proba)
            pred_label = np.argmax(proba)
            
            # Authorization logic
            if confidence > 0.6:  # Higher threshold
                user_id = le.inverse_transform([pred_label])[0]
                st.success(f"✅ Authorized: User **{user_id}**")
                st.write(f"Confidence: {confidence:.2f}")
            else:
                st.error("❌ Access Denied: Unrecognized fingerprint")
                st.write(f"Confidence: {confidence:.2f}")
                
            # Cleanup temp file
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()