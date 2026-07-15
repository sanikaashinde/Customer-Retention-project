from pathlib import Path
import joblib
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
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "churn.csv"

MODEL_PATH = BASE_DIR / "ml" / "model.pkl"

SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"

ENCODER_PATH = BASE_DIR / "ml" / "encoders.pkl"

METRICS_PATH = BASE_DIR / "ml" / "metrics.pkl"

# Mentor Requirement
TRAIN_DATA_PATH = BASE_DIR / "data" / "train.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "test.csv"

print("=" * 60)
print("CUSTOMER CHURN MODEL TRAINING")
print("=" * 60)

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)
original_df = df.copy()

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

df.columns = df.columns.str.strip()

df = df.drop_duplicates()

print("Shape After Removing Duplicates :", df.shape)

# =========================================================
# PREPROCESSING
# =========================================================

print("\nPreprocessing dataset...")

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

if "customerID" in categorical_columns:
    categorical_columns.remove("customerID")

if "Churn" in categorical_columns:
    categorical_columns.remove("Churn")

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

print("Handling missing values...")

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

for column in categorical_columns:
    df[column] = df[column].fillna(
        "missing"
    )

print("Missing Values After Cleaning")

print(df.isnull().sum())

# =========================================================
# TARGET COLUMN
# =========================================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget Distribution")

print(df["Churn"].value_counts())

# =========================================================
# LABEL ENCODING
# =========================================================

print("\nEncoding categorical features...")

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(
        df[column]
    )

    encoders[column] = encoder

print("Categorical Encoding Completed")

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop(
    columns=["customerID", "Churn"]
)

y = df["Churn"]

print("\nFeature Matrix Shape :", X.shape)
print("Target Shape :", y.shape)

print("\nFeature Columns")

for column in X.columns:
    print("-", column)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# =========================================================
# SAVE TRAIN & TEST DATASETS (Mentor Requirement)
# =========================================================

print("\nSaving reusable train/test datasets...")

train_original = original_df.loc[X_train.index].copy()
test_original = original_df.loc[X_test.index].copy()

train_original.to_csv(TRAIN_DATA_PATH, index=False)
test_original.to_csv(TEST_DATA_PATH, index=False)

print("Train dataset saved :", TRAIN_DATA_PATH)
print("Test dataset saved  :", TEST_DATA_PATH)

# =========================================================
# FEATURE SCALING
# =========================================================

print("\nScaling numeric features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling Completed")

# =========================================================
# MODEL TRAINING
# =========================================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42,
    class_weight="balanced"
)

model.fit(
    X_train_scaled,
    y_train
)

print("Model Training Completed Successfully")

# =========================================================
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(
    X_test_scaled
)[:, 1]

print("Prediction Completed")

# =========================================================
# MODEL EVALUATION
# =========================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred
))

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")

print(
    importance.head(10)
)

# =========================================================
# METRICS DICTIONARY
# =========================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,

    # Keep both keys for compatibility with all dashboard pages
    "f1": f1,
    "f1_score": f1,

    "roc_auc": roc_auc,
    "confusion_matrix": cm,
    "classification_report": report,
    "feature_importance": importance
}

print("\nEvaluation Completed Successfully")

# =========================================================
# SAVE MODEL ARTIFACTS
# =========================================================

print("\nSaving model artifacts...")

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    encoders,
    ENCODER_PATH
)

joblib.dump(
    metrics,
    METRICS_PATH
)

print("Artifacts saved successfully.")

print(f"Model    : {MODEL_PATH}")
print(f"Scaler   : {SCALER_PATH}")
print(f"Encoders : {ENCODER_PATH}")
print(f"Metrics  : {METRICS_PATH}")

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)