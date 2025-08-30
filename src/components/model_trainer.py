# model_trainer.py

import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import joblib

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

class ModelTrainer:
    def __init__(self, best_params=None, n_estimators=1000, recall_target=0.98):
        """
        Initialize ModelTrainer with parameters.

        Args:
            best_params (dict): Tuned hyperparameters for LightGBM.
            n_estimators (int): Max number of boosting iterations.
            recall_target (float): Minimum recall requirement.
        """
        self.best_params = best_params or {}
        self.n_estimators = n_estimators
        self.recall_target = recall_target
        self.model = None
        self.best_threshold = 0.5  # default

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train LightGBM model with early stopping.
        """
        print("Training LightGBM model...")
        self.model = lgb.LGBMClassifier(**self.best_params, n_estimators=self.n_estimators)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(50)
            ]
        )

        # Find optimal threshold
        self._find_best_threshold(X_val, y_val)

    def _find_best_threshold(self, X_val, y_val):
        """
        Find best threshold based on recall >= target and max precision.
        """
        print("\nFinding best threshold...")
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)

        valid_idxs = np.where(recalls[:-1] >= self.recall_target)[0]

        if len(valid_idxs) > 0:
            best_idx = valid_idxs[np.argmax(precisions[valid_idxs])]
            self.best_threshold = thresholds[best_idx]
            print(f"Best threshold found: {self.best_threshold:.4f}")
        else:
            self.best_threshold = 0.5
            print(f"No threshold found with recall >= {self.recall_target}. Using default 0.5.")


    def evaluate(self, X_val, y_val):
        """
        Evaluate model using the chosen threshold.
        """
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, digits=4))

    def predict(self, X):
        """
        Predict with custom threshold.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        return (y_pred_proba >= self.best_threshold).astype(int)

    def save_model(self, filepath="final_model.pkl"):
        """
        Save model and threshold to disk.
        """
        joblib.dump({"model": self.model, "threshold": self.best_threshold}, filepath)
        print(f"Model saved at {filepath}")

    def load_model(self, filepath="final_model.pkl"):
        """
        Load model and threshold from disk.
        """
        data = joblib.load(filepath)
        self.model = data["model"]
        self.best_threshold = data["threshold"]
        print(f"Model loaded from {filepath}")


from model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation, DataTransformationConfig



