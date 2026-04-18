from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from model import RiskModel

app = FastAPI(title="Risk Prediction API")

@app.on_event("startup")
def load_model():
    app.state.risk_model = RiskModel("risk_model.joblib")
    print("MODEL LOADED:", app.state.risk_model.model)

class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: PredictionInput, request: Request):
    risk_model = request.app.state.risk_model
    features = [data.feature1, data.feature2, data.feature3]

    if risk_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = risk_model.predict(features)
    return {"prediction": int(result)}

@app.get("/health")
def health():
    return {"status": "ok"}