from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import sqlite3

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Customer Churn Prediction")

st.write(
    "Select a customer and predict whether the customer is likely to churn."
)

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "churn.csv"
MODEL_PATH = BASE_DIR / "ml" / "model.pkl"
SCALER_PATH = BASE_DIR / "ml" / "scaler.pkl"
ENCODER_PATH = BASE_DIR / "ml" / "encoders.pkl"
DB_PATH = BASE_DIR / "database" / "customer.db"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)

    return model, scaler, encoders


@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


model, scaler, encoders = load_model()

df = load_data()

# =====================================================
# CUSTOMER SELECT
# =====================================================

customer_id = st.selectbox(
    "Select Customer ID",
    sorted(df["customerID"].tolist())
)

customer = df[df["customerID"] == customer_id].iloc[0]

# =====================================================
# CUSTOMER DETAILS
# =====================================================

st.subheader("👤 Customer Details")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric("Customer ID", customer["customerID"])
    st.metric("Gender", customer["gender"])

with c2:

    st.metric("Tenure", customer["tenure"])
    st.metric("Contract", customer["Contract"])

with c3:

    st.metric("Monthly Charges", f"₹ {customer['MonthlyCharges']}")
    st.metric("Total Charges", f"₹ {customer['TotalCharges']}")

st.markdown("---")

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("🚀 Predict Churn", use_container_width=True):

    input_df = customer.to_frame().T.copy()

    input_df.drop(
        columns=["customerID", "Churn"],
        inplace=True
    )

    input_df["TotalCharges"] = pd.to_numeric(
        input_df["TotalCharges"],
        errors="coerce"
    )

    input_df["TotalCharges"] = input_df["TotalCharges"].fillna(
        input_df["TotalCharges"].median()
    )

    # Encode categorical columns

    for col, encoder in encoders.items():

        if col in input_df.columns:

            input_df[col] = encoder.transform(
                input_df[col]
            )

    # Scale

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    # =====================================================
    # Risk Level
    # =====================================================

    if probability >= 0.80:

        risk = "High"

    elif probability >= 0.50:

        risk = "Medium"

    else:

        risk = "Low"

    # =====================================================
    # DATABASE
    # =====================================================

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_history'"
    )

    table_exists = cursor.fetchone()

      # =====================================================
    # DATABASE SCHEMA CHECK
    # =====================================================

    if table_exists:

        cursor.execute("PRAGMA table_info(prediction_history)")
        columns = [col[1] for col in cursor.fetchall()]

        if "customerID" not in columns:

            cursor.execute("DROP TABLE prediction_history")

            cursor.execute("""
            CREATE TABLE prediction_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customerID TEXT,
                prediction TEXT,
                probability REAL,
                risk TEXT,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()

    else:

        cursor.execute("""
        CREATE TABLE prediction_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT,
            prediction TEXT,
            probability REAL,
            risk TEXT,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()

    # =====================================================
    # SAVE PREDICTION
    # =====================================================

    cursor.execute(
        """
        INSERT INTO prediction_history
        (customerID, prediction, probability, risk)
        VALUES (?, ?, ?, ?)
        """,
        (
            customer_id,
            "Churn" if prediction == 1 else "No Churn",
            float(probability),
            risk
        )
    )

    conn.commit()
    conn.close()

    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("📈 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:

        st.error("❌ Customer Will Churn")

    else:

        st.success("✅ Customer Will Stay")

    # =====================================================
    # RISK ANALYSIS
    # =====================================================

    st.markdown("---")

    st.subheader("⚠️ Risk Analysis")

    if risk == "High":

        st.error("🔴 High Risk Customer")

        st.warning(
            "Offer retention campaigns, discounts and proactive customer support immediately."
        )

    elif risk == "Medium":

        st.warning("🟠 Medium Risk Customer")

        st.info(
            "Recommend promotional offers and periodic follow-up communication."
        )

    else:

        st.success("🟢 Low Risk Customer")

        st.info(
            "Customer appears loyal. Continue regular engagement and reward programs."
        )

    # =====================================================
    # ACTUAL LABEL
    # =====================================================

    st.markdown("---")

    st.subheader("📋 Actual Dataset Label")

    if customer["Churn"] == "Yes":

        st.error("Actual Label: Customer Churned")

    else:

        st.success("Actual Label: Customer Did Not Churn")  
