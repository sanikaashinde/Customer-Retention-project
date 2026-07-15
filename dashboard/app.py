import streamlit as st
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
METRICS_PATH = BASE_DIR / "ml" / "metrics.pkl"

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_metrics():
    """
    Load evaluation metrics from metrics.pkl.
    Returns a dictionary if successful, otherwise None.
    """
    if not METRICS_PATH.exists():
        st.error(f"Metrics file not found:\n{METRICS_PATH}")
        return None

    try:
        return joblib.load(METRICS_PATH)
    except Exception as e:
        st.error(f"Unable to load metrics.\n\nError: {e}")
        return None


def metric_value(metrics, key, fallback_key=None):
    """
    Safely return a metric value.
    """
    if metrics is None:
        return 0.0

    value = metrics.get(key)

    if value is None and fallback_key:
        value = metrics.get(fallback_key)

    if value is None:
        return 0.0

    return float(value)


# =====================================================
# LOAD METRICS
# =====================================================

metrics = load_metrics()

# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Prediction using Machine Learning")

st.markdown("""
This project predicts whether a telecom customer is likely to **churn (leave the service)** using a
**Random Forest Machine Learning model** trained on the **IBM Telco Customer Churn Dataset**.

The application provides Customer Churn Prediction, Prediction History,
Business Insights, and Model Performance through an interactive Streamlit dashboard.
""")

st.divider()

# =====================================================
# PROJECT WORKFLOW
# =====================================================

st.header("Project Workflow")

st.code("""
IBM Telco Dataset
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
Customer Churn Prediction
        ↓
Business Insights Dashboard
""")

st.divider()

# =====================================================
# DATASET & MODEL
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Dataset Information")

    st.markdown("""
- **Dataset:** IBM Telco Customer Churn
- **Records:** 7,043 Customers
- **Features:** 20
- **Target Variable:** Churn
- **Missing Values:** Handled during preprocessing
""")

with col2:

    st.subheader("Machine Learning Model")

    st.markdown("""
- **Algorithm:** Random Forest Classifier
- **Train-Test Split:** 80 : 20
- **Missing Value Handling**
- **Label Encoding**
- **Feature Scaling**
""")

st.divider()

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.header("Model Performance")

if metrics is not None:

    accuracy = metric_value(metrics, "accuracy")
    precision = metric_value(metrics, "precision")
    recall = metric_value(metrics, "recall")
    f1 = metric_value(metrics, "f1_score", "f1")
    roc_auc = metric_value(metrics, "roc_auc", "auc")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Precision", f"{precision * 100:.2f}%")
    c3.metric("Recall", f"{recall * 100:.2f}%")
    c4.metric("F1 Score", f"{f1 * 100:.2f}%")
    c5.metric("ROC AUC", f"{roc_auc * 100:.2f}%")

else:
    st.warning("Model metrics are unavailable.")

st.divider()

# =====================================================
# PROJECT ARCHITECTURE
# =====================================================

st.header("Project Architecture")

st.code("""
data/
├── churn.csv
├── train.csv
└── test.csv
        │
        ▼
Data Preprocessing
        │
        ▼
Random Forest Model
        │
        ▼
ml/
├── model.pkl
├── scaler.pkl
├── encoders.pkl
└── metrics.pkl
        │
        ▼
Streamlit Dashboard
        │
        ├── Dashboard
        ├── Churn Prediction
        ├── Prediction History
        └── Model Performance
""")

st.divider()

# =====================================================
# DASHBOARD MODULES
# =====================================================

st.header("Dashboard Modules")

st.markdown("""
- Dashboard
- Churn Prediction
- Prediction History
- Model Performance
""")

st.divider()

# =====================================================
# BUSINESS OBJECTIVE
# =====================================================

st.header("Business Objective")

st.info("""
The objective of this project is to identify customers who are likely to churn,
allowing telecom companies to take proactive retention actions such as personalized
offers, loyalty rewards, and customer engagement strategies.
""")

st.divider()

st.success("End-to-End Customer Churn Prediction System using Machine Learning and Streamlit")