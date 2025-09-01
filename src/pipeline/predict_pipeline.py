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

            # Feature extraction only
            features = self.transformer.feature_extraction(df)

            # Make sure 'label' never exists at inference
            if "label" in features.columns:
                features = features.drop(columns=["label"])


            # Predict probability
            y_pred_proba = self.model.predict_proba(features)[:, 1]

            # Apply threshold
            prediction = (y_pred_proba >= self.threshold).astype(int)[0]

            # Map numeric to label
            label = "phishing" if prediction == 1 else "benign"

            return {"prediction": label, "probability": float(y_pred_proba[0])}

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        predictor = PredictPipeline()
        test_url = "http://login-update-verification1234.com/paypal"
        result = predictor.predict_single_url(test_url)
        print(f"URL: {test_url}")
        print(f"Prediction: {result['prediction']}, Probability: {result['probability']:.4f}")
    except Exception as e:
        print(e)
# python -m src.pipeline.predict_pipeline