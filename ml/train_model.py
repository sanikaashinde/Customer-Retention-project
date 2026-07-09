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

try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    df = pd.read_csv(DATA_PATH)

except Exception as e:
    st.error(f"Error loading files:\n\n{e}")
    st.stop()

# =====================================================
# VERIFY METRICS FILE
# =====================================================

st.caption("Metrics loaded from ml/metrics.pkl")

# Uncomment this if you want to debug
# st.write(metrics)

# =====================================================
# METRICS
# =====================================================

accuracy = metrics["accuracy"]
precision = metrics["precision"]
recall = metrics["recall"]
f1 = metrics["f1"]
cm = metrics["confusion_matrix"]

st.subheader("Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

col2.metric(
    "Precision",
    f"{precision*100:.2f}%"
)

col3.metric(
    "Recall",
    f"{recall*100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{f1*100:.2f}%"
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

X = df.drop(columns=["customerID", "Churn"])

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
    title="Top Features Influencing Churn"
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

st.subheader("Feature Importance Table")

importance["Importance"] = importance["Importance"].round(4)

st.dataframe(
    importance,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

st.success("Model performance shown above is based on the test dataset used during training.")