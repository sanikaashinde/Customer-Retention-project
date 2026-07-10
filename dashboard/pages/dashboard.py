from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Dashboard")
st.markdown("Business Analytics Dashboard using Telco Customer Churn Dataset")

# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "churn.csv"

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# =====================================================
# KPI CARDS
# =====================================================

total_customers = len(df)

churn_customers = (df["Churn"] == "Yes").sum()

active_customers = (df["Churn"] == "No").sum()

churn_rate = churn_customers / total_customers * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    total_customers
)

col2.metric(
    "Active Customers",
    active_customers
)

col3.metric(
    "Churn Customers",
    churn_customers
)

col4.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

st.markdown("---")

# =====================================================
# CHARTS
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.pie(
        df,
        names="Churn",
        title="Customer Churn Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.histogram(
        df,
        x="Contract",
        color="Churn",
        barmode="group",
        title="Contract Type vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# SECOND ROW
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="InternetService",
        color="Churn",
        barmode="group",
        title="Internet Service vs Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# THIRD ROW
# =====================================================

col1, col2 = st.columns(2)

with col1:

    fig = px.histogram(
        df,
        x="tenure",
        color="Churn",
        nbins=30,
        title="Customer Tenure Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with col2:

    payment = (
        df["PaymentMethod"]
        .value_counts()
        .reset_index()
    )

    payment.columns = [
        "Payment Method",
        "Customers"
    ]

    fig = px.bar(
        payment,
        x="Payment Method",
        y="Customers",
        color="Customers",
        title="Payment Method Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# DATASET
# =====================================================

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =====================================================
# DOWNLOAD
# =====================================================

st.download_button(
    label="📥 Download Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="churn_dataset.csv",
    mime="text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction System | Streamlit + Machine Learning + Python"
)

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

st.markdown("---")
st.header("📊 Business Insights")

# Highest Churn Contract
contract_churn = (
    df.groupby("Contract")["Churn"]
      .apply(lambda x: (x == "Yes").mean() * 100)
)

highest_contract = contract_churn.idxmax()

# Fiber Users Churn %
fiber_df = df[df["InternetService"] == "Fiber optic"]

fiber_churn = (
    (fiber_df["Churn"] == "Yes").mean() * 100
)

# Highest Monthly Charges among churned customers
highest_monthly = (
    df[df["Churn"] == "Yes"]["MonthlyCharges"].max()
)

# Most Loyal Customers
loyal_customers = (
    df[df["Churn"] == "No"]["tenure"].max()
)

# Top Churn Risk Group
risk_group = (
    df[df["Churn"] == "Yes"]
    .groupby("Contract")
    .size()
    .idxmax()
)

st.success(
    f"🔴 **Top Churn Risk Group:** Customers with **{risk_group}** contracts have the highest churn volume."
)

st.warning(
    f"📄 **Highest Churn Contract:** **{highest_contract}** ({contract_churn.max():.1f}% churn rate)."
)

st.info(
    f"💰 **Highest Monthly Charges (Churned Customers): ₹{highest_monthly:.2f}**"
)

st.success(
    f"⭐ **Most Loyal Customers:** Customers staying for **{loyal_customers} months**."
)

st.error(
    f"🌐 **Fiber Optic Users Churn Rate:** {fiber_churn:.1f}%"
)

st.markdown("---")
st.subheader("💡 Business Recommendations")

st.markdown("""
- Retain customers on **Month-to-month contracts** with personalized offers.
- Provide discounts for customers with **high monthly charges**.
- Improve customer support for **Fiber optic users**.
- Encourage long-term contracts through loyalty rewards.
- Target high-risk customers before contract renewal.
""")
