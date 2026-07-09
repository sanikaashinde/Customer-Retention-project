from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Performance")

st.write(
    "Performance metrics of the trained Random Forest model evaluated on the test dataset."
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
METRICS_PATH = BASE_DIR / "ml" / "metrics.pkl"
DATA_PATH = BASE_DIR / "data" / "churn.csv"

# =====================================================
# LOAD FILES
# =====================================================

model = joblib.load(MODEL_PATH)
metrics = joblib.load(METRICS_PATH)
df = pd.read_csv(DATA_PATH)

# Remove columns not used for feature importance
X = df.drop(columns=["customerID", "Churn"])

# =====================================================
# METRICS
# =====================================================

accuracy = metrics["accuracy"]
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]
cm = metrics["confusion_matrix"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy:.2%}"
)

col2.metric(
    "Precision",
    f"{precision:.2%}"
)

col3.metric(
    "Recall",
    f"{recall:.2%}"
)

col4.metric(
    "F1 Score",
    f"{f1:.2%}"
)

st.markdown("---")

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("Confusion Matrix")

cm_df = pd.DataFrame(
    cm,
    index=["Actual No Churn", "Actual Churn"],
    columns=["Predicted No Churn", "Predicted Churn"]
)

st.dataframe(
    cm_df,
    use_container_width=True,
    hide_index=False
)

st.markdown("---")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader("Feature Importance")

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
    title="Feature Importance",
    text_auto=".3f"
)

fig.update_layout(
    yaxis=dict(categoryorder="total ascending"),
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

# =====================================================
# TOP IMPORTANT FEATURES
# =====================================================

st.subheader("Top Important Features")

st.dataframe(
    importance.reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =====================================================
# MODEL SUMMARY
# =====================================================

st.subheader("Model Summary")

summary = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Training Samples",
        "Testing Samples",
        "Number of Features",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        "Random Forest Classifier",
        5634,
        1409,
        len(X.columns),
        f"{accuracy:.2%}",
        f"{precision:.2%}",
        f"{recall:.2%}",
        f"{f1:.2%}"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)