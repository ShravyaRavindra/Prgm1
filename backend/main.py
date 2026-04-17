from fastapi import FastAPI
from pydantic import BaseModel
from model import RiskModel

app = FastAPI(title="Risk Prediction API")
risk_model = RiskModel()

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    features = [data.feature1, data.feature2, data.feature3]
    result = risk_model.predict(features)
    return {"prediction": result}
@app.get("/health")
def health():
    return {"status": "ok"}
