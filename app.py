from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

# Load the model
model = joblib.load("network_issue_detector.pkl")  # Ensure network_issue_detector.pkl is present

# Initialize FastAPI app
app = FastAPI(title="Network Performance Predictor")

# Input schema
class InputFeatures(BaseModel):
    latency_ms: float
    packet_loss_pct: float
    jitter_ms: float
    bandwidth_usage_pct: float

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "API is running"}

# Endpoint to get expected input format
@app.get("/format")
def get_format():
    return {
        "expected_format": {
            "latency_ms": "float (e.g., 45.2)",
            "packet_loss_pct": "float (e.g., 0.3)",
            "jitter_ms": "float (e.g., 4.5)",
            "bandwidth_usage_pct": "float (e.g., 68.0)"
        }
    }

# Prediction endpoint
@app.post("/predict")
def predict(input: InputFeatures):
    data = pd.DataFrame([{
        'latency_ms': input.latency_ms,
        'packet_loss_%': input.packet_loss_pct,
        'jitter_ms': input.jitter_ms,
        'bandwidth_usage_%': input.bandwidth_usage_pct
    }])
    
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}