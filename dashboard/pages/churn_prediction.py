import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load model
model = joblib.load("../ml/model.pkl")
scaler = joblib.load("../ml/scaler.pkl")

# Load customer data
rfm = pd.read_csv("../data/customer_rfm_segments.csv")

st.title("🤖 Customer Churn Prediction")

st.caption(
    f"Prediction Time: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
)

st.write("Enter Customer ID to predict churn.")

customer_id = st.number_input(
    "Customer ID",
    min_value=0.0,
    step=1.0
)

if st.button("Predict Customer Churn"):

    customer = rfm[
        rfm["Customer ID"] == customer_id
    ]

    if customer.empty:

        st.error("Customer ID not found.")

    else:

        recency = customer.iloc[0]["Recency"]
        frequency = customer.iloc[0]["Frequency"]
        monetary = customer.iloc[0]["Monetary"]
        segment = customer.iloc[0]["Segment"]

        input_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1]

        st.subheader("Customer Details")

        st.write("Customer ID:", int(customer_id))
        st.write("Segment:", segment)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Recency",
                recency
            )

        with col2:
            st.metric(
                "Frequency",
                frequency
            )

        with col3:
            st.metric(
                "Monetary",
                round(monetary, 2)
            )

        st.subheader("Churn Probability")

        st.progress(float(probability))

        st.write(
            f"Risk Probability: {round(probability*100,2)}%"
        )

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "❌ Customer Will Churn"
            )

        else:

            st.success(
                "✅ Customer Will Stay"
            )

        if probability > 0.8:

            st.error(
                "🔴 High Risk Customer"
            )

            st.warning(
                "Recommended Action: Provide discount or retention campaign."
            )

        elif probability > 0.5:

            st.warning(
                "🟠 Medium Risk Customer"
            )

            st.info(
                "Recommended Action: Send promotional campaign."
            )

        else:

            st.success(
                "🟢 Low Risk Customer"
            )

            st.info(
                "Customer is loyal and active."
            )