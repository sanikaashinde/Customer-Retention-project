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
# LOAD FILES
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
# CUSTOMER SELECTION
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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customer ID", customer["customerID"])
    st.metric("Gender", customer["gender"])

with col2:
    st.metric("Tenure", customer["tenure"])
    st.metric("Contract", customer["Contract"])

with col3:
    st.metric("Monthly Charges", f"₹{customer['MonthlyCharges']}")
    st.metric("Total Charges", f"₹{customer['TotalCharges']}")

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

    # Prediction
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
    # SAVE TO DATABASE
    # =====================================================

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customerID TEXT,
        prediction TEXT,
        probability REAL,
        risk TEXT,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute(
        """
        INSERT INTO predictions
        (customerID,prediction,probability,risk)
        VALUES (?,?,?,?)
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
        f"{probability*100:.2f}%"
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
            "Recommendation: Offer discounts, loyalty rewards and retention campaigns immediately."
        )

    elif risk == "Medium":

        st.warning("🟠 Medium Risk Customer")

        st.info(
            "Recommendation: Send promotional offers and follow-up communication."
        )

    else:

        st.success("🟢 Low Risk Customer")

        st.info(
            "Recommendation: Customer is loyal. Continue regular engagement."
        )

    # =====================================================
    # ACTUAL LABEL
    # =====================================================

    st.markdown("---")

    st.subheader("📋 Actual Dataset Label")

    if customer["Churn"] == "Yes":

        st.error("Actual Label : Customer Churned")

    else:

        st.success("Actual Label : Customer Did Not Churn")