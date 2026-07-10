import joblib
import pandas as pd

model = joblib.load("ml/model.pkl")
scaler = joblib.load("ml/scaler.pkl")
encoders = joblib.load("ml/encoders.pkl")

print("✅ Model Loaded")
print("✅ Scaler Loaded")
print("✅ Encoders Loaded")

df = pd.read_csv("data/churn.csv")

sample = df.iloc[[0]].copy()

sample.drop(columns=["customerID", "Churn"], inplace=True)

sample["TotalCharges"] = pd.to_numeric(
    sample["TotalCharges"],
    errors="coerce"
)

sample.fillna(sample.median(numeric_only=True), inplace=True)

for col, encoder in encoders.items():
    if col in sample.columns:
        sample[col] = encoder.transform(sample[col])

sample = scaler.transform(sample)

prediction = model.predict(sample)[0]

print("Prediction:", prediction)

print("\n✅ Smoke Test Passed")
