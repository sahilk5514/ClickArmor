import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize pipeline
predictor = PredictPipeline()

# Page config
st.set_page_config(page_title="ClickArmor", page_icon="🛡️", layout="centered")

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .stTextInput input {
            border-radius: 12px;
            border: 1px solid #cccccc;
            padding: 10px;
            font-size: 16px;
        }
        .stButton button {
            border-radius: 12px;
            font-size: 16px;
            padding: 8px 24px;
            background-color: #4CAF50;
            color: white;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 style='text-align: center;'>🛡️ ClickArmor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Check if a URL is <b>benign</b> ✅ or <b>phishing</b> 🚨</p>", unsafe_allow_html=True)

# Input URL
url = st.text_input("🔗 Enter a URL to analyze:", placeholder="e.g. https://paypal-login-secure-update.com")

# Prediction
if st.button("🔍 Analyze URL"):
    if url.strip() == "":
        st.warning("⚠️ Please enter a valid URL.")
    else:
        result = predictor.predict_single_url(url)
        label = result["prediction"]
        probability = result["probability"]

        if label == "phishing":
            st.markdown(
                f"<div style='background-color:#ffe6e6; padding:15px; border-radius:12px; text-align:center;'>"
                f"<h3 style='color:#d9534f;'>🚨 Prediction: Phishing</h3>"
                f"<p>Confidence: <b>{probability:.4f}</b></p>"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:#e6ffe6; padding:15px; border-radius:12px; text-align:center;'>"
                f"<h3 style='color:#28a745;'>✅ Prediction: Benign</h3>"
                f"<p>Confidence: <b>{probability:.4f}</b></p>"
                f"</div>", unsafe_allow_html=True
            )
