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
TEST_DATA_PATH = BASE_DIR / "data" / "test.csv"

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

full_df = load_data()

# =====================================================
# DATA SOURCE SELECTION
# =====================================================

st.subheader("📂 Select Data Source")

source = st.radio(
    "Choose Dataset",
    (
        "Full Dataset",
        "Test Dataset Only"
    ),
    horizontal=True
)

if source == "Full Dataset":

    df = full_df

else:

    if TEST_DATA_PATH.exists():

        df = pd.read_csv(TEST_DATA_PATH)

    else:

        st.error(
            "test.csv not found. Please run ml/train_model.py first."
        )
        st.stop()

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

c1, c2, c3 = st.columns(3)

with c1:

    st.metric("Customer ID", customer["customerID"])
    st.metric("Gender", customer["gender"])

with c2:

    st.metric("Tenure", customer["tenure"])
    st.metric("Contract", customer["Contract"])

with c3:

    st.metric(
        "Monthly Charges",
        f"₹ {customer['MonthlyCharges']}"
    )

    st.metric(
        "Total Charges",
        f"₹ {customer['TotalCharges']}"
    )

st.markdown("---")

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button(
    "🚀 Predict Churn",
    use_container_width=True
):
   
    # =====================================================
    # PREPARE INPUT
    # =====================================================

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

    # =====================================================
    # ENCODE CATEGORICAL FEATURES
    # =====================================================

    for col, encoder in encoders.items():

        if col in input_df.columns:

            # Encode only if values are still strings
            if input_df[col].dtype == object:

                input_df[col] = encoder.transform(
                    input_df[col]
                )

    # =====================================================
    # SCALE FEATURES
    # =====================================================

    input_scaled = scaler.transform(input_df)

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    prediction_text = (
        "Churn"
        if prediction == 1
        else "No Churn"
    )

    # =====================================================
    # RISK LEVEL
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customerID TEXT,
        prediction TEXT,
        probability REAL,
        risk TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()

    cursor.execute(
        """
        INSERT INTO prediction_history
        (
            customerID,
            prediction,
            probability,
            risk
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            customer_id,
            prediction_text,
            float(probability),
            risk
        )
    )

    conn.commit()
    conn.close()

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader("📈 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error("❌ Customer Will Churn")

        else:

            st.success("✅ Customer Will Stay")

    with result_col2:

        st.metric(
            "Risk Level",
            risk
        )

    # =====================================================
    # RISK ANALYSIS
    # =====================================================

    st.markdown("---")

    st.subheader("⚠️ Risk Analysis")

    if risk == "High":

        st.error("🔴 High Risk Customer")

        st.warning(
            """
            Recommended Actions

            • Contact customer immediately
            • Offer special discount
            • Assign retention executive
            • Provide personalized support
            """
        )

    elif risk == "Medium":

        st.warning("🟠 Medium Risk Customer")

        st.info(
            """
            Recommended Actions

            • Send promotional offers
            • Follow up via email/SMS
            • Recommend suitable plans
            """
        )

    else:

        st.success("🟢 Low Risk Customer")

        st.info(
            """
            Recommended Actions

            • Customer is loyal
            • Continue engagement
            • Reward through loyalty programs
            """
        )

    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    summary = pd.DataFrame(
        {
            "Field": [
                "Customer ID",
                "Prediction",
                "Probability",
                "Risk Level"
            ],
            "Value": [
                customer_id,
                prediction_text,
                f"{probability*100:.2f}%",
                risk
            ]
        }
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # =====================================================
    # ACTUAL LABEL
    # =====================================================

    st.markdown("---")

    st.subheader("📌 Actual Dataset Label")

    if customer["Churn"] == "Yes":

        st.error("Actual Label : Customer Churned")

    else:

        st.success("Actual Label : Customer Did Not Churn")

    # =====================================================
    # MODEL CONFIDENCE
    # =====================================================

    st.markdown("---")

    st.subheader("🎯 Model Confidence")

    st.info(
        f"The model predicts **{prediction_text}** "
        f"with **{probability*100:.2f}%** confidence."
    )