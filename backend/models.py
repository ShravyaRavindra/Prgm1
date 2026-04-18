from sqlalchemy import Column, Integer, Float
from database import Base

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    feature1 = Column(Float, nullable=False)
    feature2 = Column(Float, nullable=False)
    prediction = Column(Integer, nullable=False)