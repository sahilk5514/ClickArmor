import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize pipeline
predictor = PredictPipeline()

st.set_page_config(page_title="ClickArmor", layout="centered")

st.title("ClickArmor")
st.markdown("Enter a URL below to check if it's **benign** or **phishing**.")

# Input URL
url = st.text_input("Enter URL:")

if st.button("Check"):
    if url.strip() == "":
        st.warning("⚠️ Please enter a valid URL.")
    else:
        result = predictor.predict_single_url(url)
        label = result["prediction"]
        probability = result["probability"]

        if label == "phishing":
            st.error(f"🚨 Prediction: **Phishing** (Confidence: {probability:.4f})")
        else:
            st.success(f"✅ Prediction: **Benign** (Confidence: {probability:.4f})")
