import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("📊 Customer Churn Prediction using Machine Learning")

st.markdown("""
This project predicts whether a telecom customer is likely to **churn (leave the service)** using a
**Random Forest Machine Learning model** trained on the **IBM Telco Customer Churn Dataset**.

The application provides customer predictions, business insights, model performance,
and prediction history through an interactive Streamlit dashboard.
""")

st.markdown("---")

# =====================================================
# PROJECT WORKFLOW
# =====================================================

st.header("🔄 Project Workflow")

st.code("""
IBM Telco Dataset
        ↓
Exploratory Data Analysis (EDA)
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Random Forest Model
        ↓
Model Evaluation
        ↓
Customer Churn Prediction
        ↓
Business Insights Dashboard
""")

st.markdown("---")

# =====================================================
# DATASET & MODEL
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📂 Dataset Information")

    st.markdown("""
- **Dataset:** IBM Telco Customer Churn
- **Records:** 7,043 Customers
- **Features:** 20
- **Target Variable:** Churn
- **Missing Values:** Handled during preprocessing
    """)

with col2:

    st.subheader("🤖 Machine Learning Model")

    st.markdown("""
- **Algorithm:** Random Forest Classifier
- **Train-Test Split:** 80 : 20
- **Missing Value Handling**
- **Label Encoding**
- **Feature Scaling**
    """)

st.markdown("---")

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.header("📈 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", "79.63%")
c2.metric("Precision", "66.17%")
c3.metric("Recall", "47.18%")
c4.metric("F1 Score", "55.09%")

st.markdown("---")

# =====================================================
# PROJECT ARCHITECTURE
# =====================================================

st.header("🏗 Project Architecture")

st.code("""
data/churn.csv
       │
       ▼
EDA & Preprocessing
       │
       ▼
Random Forest Model
       │
       ▼
model.pkl
       │
       ▼
Streamlit Dashboard
       │
       ├── Dashboard
       ├── Churn Prediction
       ├── Prediction History
       └── Model Performance
""")

st.markdown("---")

# =====================================================
# PROJECT MODULES
# =====================================================

st.header("📌 Dashboard Modules")

st.markdown("""
✅ Dashboard

✅ Churn Prediction

✅ Prediction History

✅ Model Performance
""")

st.markdown("---")

# =====================================================
# BUSINESS OBJECTIVE
# =====================================================

st.header("💡 Business Objective")

st.info("""
The objective of this project is to identify customers who are likely to churn,
allowing telecom companies to take proactive retention actions such as personalized
offers, loyalty rewards, and customer engagement strategies.
""")

st.markdown("---")

st.success("✅ End-to-End Customer Churn Prediction System using Machine Learning and Streamlit")