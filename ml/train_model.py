import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib

# Load Dataset
df = pd.read_csv("data/churn.csv")

# Display first 5 rows
print(df.head())

# Display dataset shape
print("\nDataset Shape:", df.shape)

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Remove customerID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values again
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Verify missing values are removed
print("\nMissing Values After Filling:")
print(df.isnull().sum())

# Convert Churn column into 0 and 1
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nChurn Value Counts:")
print(df["Churn"].value_counts())

print("\nFirst 5 values of Churn:")
print(df["Churn"].head())

# Encode all categorical columns

categorical_columns = df.select_dtypes(include=["object"]).columns

encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

print("\nData Types After Encoding:")
print(df.dtypes)

# Separate Features and Target

X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns)

# Split the dataset into Training and Testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("\nTraining Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully!")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# -------------------------------
# Train Random Forest Model
# -------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model Training Completed Successfully!")

# -------------------------------
# Make Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# -------------------------------
# Classification Report
# -------------------------------

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save Model and Scaler
# -------------------------------

joblib.dump(model, "ml/model.pkl")
joblib.dump(scaler, "ml/scaler.pkl")
joblib.dump(encoders, "ml/encoders.pkl")

print("\n✅ Model saved successfully!")
print("✅ Scaler saved successfully!")
print("✅ Encoders saved successfully!")

joblib.dump(encoders, "ml/encoders.pkl")