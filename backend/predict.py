import joblib
import pandas as pd

model = joblib.load("../ml/model.pkl")
scaler = joblib.load("../ml/scaler.pkl")


def predict_churn(recency, frequency, monetary):

    data = pd.DataFrame({
        "Recency": [recency],
        "Frequency": [frequency],
        "Monetary": [monetary]
    })

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0][1]

    return int(prediction), float(probability)