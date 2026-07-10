from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class CustomerPrediction(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer)
    churn_probability = Column(Float)
    risk_level = Column(String)
