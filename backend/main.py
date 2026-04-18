from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from model import RiskModel
from database import engine, Base, get_db
from models import PredictionRecord

app = FastAPI()
risk_model = RiskModel("risk_model.joblib")


class PredictionInput(BaseModel):
    feature1: float
    feature2: float
   


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: PredictionInput, db: Session = Depends(get_db)):
    try:
        prediction = risk_model.predict([[data.feature1, data.feature2]])[0]

        record = PredictionRecord(
            feature1=data.feature1,
            feature2=data.feature2,
            prediction=int(prediction)
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(PredictionRecord).all()