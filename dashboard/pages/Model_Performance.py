from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance")

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "churn.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "ml" / "encoders.pkl"

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODER_PATH)

# =====================================================
# PREPROCESS
# =====================================================

X = df.drop(columns=["customerID", "Churn"]).copy()

X["TotalCharges"] = pd.to_numeric(
    X["TotalCharges"],
    errors="coerce"
)

X["TotalCharges"] = X["TotalCharges"].fillna(
    X["TotalCharges"].median()
)

for col, encoder in encoders.items():
    if col in X.columns:
        X[col] = encoder.transform(X[col])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X_scaled = scaler.transform(X)

y_pred = model.predict(X_scaled)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Precision", f"{precision:.2%}")
c3.metric("Recall", f"{recall:.2%}")
c4.metric("F1 Score", f"{f1:.2%}")

st.markdown("---")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual No", "Actual Yes"],
    columns=["Pred No", "Pred Yes"]
)

st.subheader("Confusion Matrix")

st.dataframe(
    cm_df,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

st.subheader("Top Important Features")

st.dataframe(
    importance,
    use_container_width=True
)