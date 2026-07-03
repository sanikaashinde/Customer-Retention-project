from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict_churn
import sqlite3
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    "database",
    "customer.db"
)

# 👇 DB_PATH define zhalyanantarach database connection
conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id REAL,
    recency REAL,
    frequency REAL,
    monetary REAL,
    probability REAL,
    risk_level TEXT,
    prediction TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

class Customer(BaseModel):
    customer_id: float
    recency: float
    frequency: float
    monetary: float


@app.get("/")
def home():
    return {
        "message": "Customer Retention API Running"
    }


@app.post("/predict")
def predict(customer: Customer):

    prediction, probability = predict_churn(
        customer.recency,
        customer.frequency,
        customer.monetary
    )

    risk = "Low"

    if probability > 0.70:
        risk = "High"
    elif probability > 0.40:
        risk = "Medium"

    prediction_text = ("Churn" if prediction == 1 else "Stay")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (
            customer_id,
            recency,
            frequency,
            monetary,
            probability,
            risk_level,
            prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            customer.customer_id,
            customer.recency,
            customer.frequency,
            customer.monetary,
            probability,
            risk,
            prediction_text
        )
    )

    conn.commit()
    conn.close()

    if risk == "High":
        recommendation = """
    🚨 High churn risk detected.

    • Send a discount coupon.
    • Provide retention offers.
    • Contact customer personally.
    • Launch win-back campaign.
    """

    elif risk == "Medium":
        recommendation = """
    ⚠ Moderate churn risk.

    • Send promotional offers.
    • Recommend popular products.
    • Offer loyalty points.
    • Run email campaigns.
    """

    else:
        recommendation = """
    ✅ Healthy customer.

    • Enroll in loyalty program.
    • Offer premium membership.
    • Upsell products.
    • Reward repeat purchases.
    """
        
    return {
        "prediction": prediction_text,
        "churn_probability": round(probability * 100, 2),
        "risk_level": risk,
        "recommendation": recommendation
    }