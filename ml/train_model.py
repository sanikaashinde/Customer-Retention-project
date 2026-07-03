import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
from sklearn.preprocessing import StandardScaler

# Dataset load
df = pd.read_csv("data/customer_rfm_segments.csv")

# Create Churn column
# 1 = Customer may churn
# 0 = Customer stays

df["Churn"] = df["Segment"].apply(
    lambda x: 1 if x in ["Lost Customers", "At Risk"] else 0
)

# Check how many customers churned or stayed
print("Churn Distribution:")
print(df["Churn"].value_counts())

# Show first 5 rows
print("\nUpdated Dataset:")
print(df.head())

# Features and Target

X = df[[
    "Recency",
    "Frequency",
    "Monetary",
]]

y = df["Churn"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training

model = RandomForestClassifier(random_state=42)

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed Successfully!")

# Save Model

joblib.dump(model, "models/churn_model.pkl")

import os


# Create ml folder if it doesn't exist
os.makedirs("ml", exist_ok=True)

joblib.dump(model, "ml/model.pkl")
joblib.dump(scaler, "ml/scaler.pkl")

print("Current Working Directory:", os.getcwd())
print("Model exists:", os.path.exists("ml/model.pkl"))
print("Scaler exists:", os.path.exists("ml/scaler.pkl"))

print("Current Working Directory:", os.getcwd())
print("Model exists:", os.path.exists("models/churn_model.pkl"))
print("Scaler exists:", os.path.exists("models/scaler.pkl"))
print("Scaler size:", os.path.getsize("models/scaler.pkl"))

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel Saved Successfully!")
# Predictions

y_pred = model.predict(X_test_scaled)

# Evaluation Metrics

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))