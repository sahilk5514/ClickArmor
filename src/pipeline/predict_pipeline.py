# src/pipeline/predict_pipeline.py

import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation


class PredictPipeline:
    def __init__(self, model_path="artifacts/lightgbm_model.pkl"):
        try:
            self.model_data = load_object(model_path)
            self.model = self.model_data["model"]
            self.threshold = self.model_data.get("threshold", 0.76)
            self.transformer = DataTransformation()
        except Exception as e:
            raise CustomException(e, sys)

def predict_single_url(self, url: str):
    try:
        # Convert URL to DataFrame
        df = pd.DataFrame({"url": [url]})

        # Extract features
        features = self.transformer.feature_extraction(df)

        # Drop label if exists
        if "label" in features.columns:
            features = features.drop(columns=["label"])

        # Predict probabilities: [benign_prob, phishing_prob]
        y_pred_proba = self.model.predict_proba(features)[0]  # single row

        # Decide class based on threshold
        is_phishing = int(y_pred_proba[1] >= self.threshold)

        # Map numeric prediction to label
        label = "phishing" if is_phishing == 1 else "benign"

        # Confidence = probability of predicted class in percentage
        confidence = y_pred_proba[is_phishing] * 100  # convert to %

        return {
            "prediction": label,
            "confidence": round(confidence, 2)  # e.g., 92.34%
        }

    except Exception as e:
        print("Error in predict_single_url:", str(e))
        import traceback
        traceback.print_exc()
        raise




# if __name__ == "__main__":
#     try:
#         predictor = PredictPipeline()
#         test_url = "http://login-update-verification1234.com/paypal"
#         result = predictor.predict_single_url(test_url)
#         print(f"URL: {test_url}")
#         print(f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
#     except Exception as e:
#         print(e)
# # python -m src.pipeline.predict_pipeline