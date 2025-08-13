from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import extract_features
import pickle
import os

# Load model
model_path = os.path.join("model", "phishing_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# FastAPI app instance
app = FastAPI()

# Request schema
class URLRequest(BaseModel):
    url: str

# Home route
@app.get("/")
def read_root():
    return {"message": "ClickArmor API is running!"}

# Prediction route
@app.post("/predict")
def predict(request: URLRequest):
    features = extract_features(request.url)
    prediction = model.predict([features])[0]
    return {"url": request.url, "is_phishing": bool(prediction)}
