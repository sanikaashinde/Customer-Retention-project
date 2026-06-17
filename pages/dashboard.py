import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)

# ============================================================
# SESSION
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = True

# ============================================================
# HEADER
# ============================================================

st.title("🚀 Customer Intelligence Platform")
st.markdown(
    "### AI Powered Customer Segmentation, Churn Prediction & Customer Lifetime Value Analysis"
)

col1, col2 = st.columns([9, 1])

with col2:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ============================================================
# LOAD DATA
# ============================================================

import pandas as pd

df = pd.read_csv("clean_online_retail.zip", compression="zip")

df.columns = df.columns.str.strip()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

df["TotalAmount"] = (
    df["Quantity"] *
    df["Price"]
)

reference_date = df["InvoiceDate"].max()

# ============================================================
# RFM CALCULATION
# ============================================================

rfm = df.groupby(
    "Customer ID"
).agg({

    "InvoiceDate": lambda x: (
        reference_date -
        x.max()
    ).days,

    "Invoice": "count",

    "TotalAmount": "sum"

})

rfm.columns = [

    "Recency",

    "Frequency",

    "Monetary"

]

# ============================================================
# CLV
# ============================================================

rfm["CLV"] = (
    rfm["Monetary"] /
    rfm["Frequency"]
) * rfm["Frequency"]

# ============================================================
# CHURN LABEL
# ============================================================

rfm["Churn"] = (

    rfm["Recency"] >

    rfm["Recency"].mean()

).astype(int)

# ============================================================
# CUSTOMER SEGMENT
# ============================================================

def customer_segment(row):

    if (

        row["CLV"] >

        rfm["CLV"].quantile(0.75)

        and

        row["Recency"] <

        rfm["Recency"].median()

    ):

        return "VIP"

    elif row["Recency"] < 30:

        return "New Customer"

    elif row["Churn"] == 1:

        return "At Risk"

    else:

        return "Loyal"

rfm["Segment"] = rfm.apply(

    customer_segment,

    axis=1

)

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

X = rfm[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

y = rfm["Churn"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    random_state=42

)

model.fit(

    X_scaled,

    y

)

rfm["Churn_Prob"] = model.predict_proba(

    X_scaled

)[:, 1]

# ============================================================
# BUSINESS ACTIONS
# ============================================================

def business_action(row):

    if (

        row["CLV"] >

        rfm["CLV"].mean()

        and

        row["Churn_Prob"] < 0.30

    ):

        return "🎁 VIP Loyalty Program"

    elif row["Churn_Prob"] > 0.70:

        return "⚠️ Send Discount / Win-back Offer"

    elif (

        row["CLV"] <

        rfm["CLV"].mean()

        and

        row["Churn_Prob"] > 0.50

    ):

        return "🚫 No Major Marketing Spend"

    else:

        return "📈 Upsell / Cross-sell Campaign"

rfm["Business_Action"] = rfm.apply(

    business_action,

    axis=1

)

# ============================================================
# HEALTH SCORE
# ============================================================

rfm["Health_Score"] = (

    (
        1 -
        (
            rfm["Recency"] /
            rfm["Recency"].max()
        )
    ) * 40

    +

    (
        rfm["Frequency"] /
        rfm["Frequency"].max()
    ) * 30

    +

    (
        rfm["Monetary"] /
        rfm["Monetary"].max()
    ) * 30

).round(2)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.image(

    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",

    width=120

)

st.sidebar.title(
    "Customer Intelligence"
)

st.sidebar.markdown("---")

menu = st.sidebar.radio(

    "📂 Navigation",

    [

        "Overview",

        "Customer Search",

        "Segmentation",

        "Churn Analysis",

        "CLV Analysis",

        "Business Actions",

        "Top Customers",

        "High Risk Customers",

        "Monthly Sales",

        "Country Analytics",

        "Feature Importance",

        "Top Products",

        "Customer Battle Arena"

    ]

)

# ============================================================
# OVERVIEW
# ============================================================

if menu == "Overview":

    st.header(
        "📌 Executive Overview"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Customers",
        len(rfm)
    )

    c2.metric(
        "VIP",
        (
            rfm["Segment"]
            ==
            "VIP"
        ).sum()
    )

    c3.metric(
        "At Risk",
        (
            rfm["Segment"]
            ==
            "At Risk"
        ).sum()
    )

    c4.metric(
        "Avg CLV",
        round(
            rfm["CLV"].mean(),
            2
        )
    )

    c5.metric(
        "Churn %",
        round(
            rfm["Churn"].mean() * 100,
            2
        )
    )

    fig = px.scatter(

        rfm,

        x="Recency",

        y="Monetary",

        color="Segment",

        hover_name=rfm.index.astype(str),

        title="Customer Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    vip = len(
        rfm[
            rfm["Segment"] == "VIP"
        ]
    )

    high_risk = len(
        rfm[
            rfm["Churn_Prob"] > 0.70
        ]
    )

    best_country = (
        df.groupby(
            "Country"
        )["TotalAmount"]
        .sum()
        .idxmax()
    )

    st.subheader(
        "🤖 AI Business Insights"
    )

    st.success(f"""

• Total Customers : {len(rfm)}

• VIP Customers : {vip}

• High Risk Customers : {high_risk}

• Best Revenue Country : {best_country}

• Average CLV : ₹{round(rfm['CLV'].mean(),2)}

### AI Recommendation

✓ Reward VIP customers with loyalty programs.

✓ Send retention offers to high-risk customers.

✓ Focus marketing campaigns in {best_country}.

✓ Upsell medium-value loyal customers.

""")

    st.download_button(

        label="📥 Download Complete Report",

        data=rfm.to_csv().encode("utf-8"),

        file_name="Customer_Intelligence_Report.csv",

        mime="text/csv"

    )

# ============================================================
# CUSTOMER SEARCH
# ============================================================

elif menu == "Customer Search":

    st.header(
        "🔍 Customer Search"
    )

    cid = st.text_input(
        "Enter Customer ID"
    )

    if cid:

        try:

            customer = rfm[
                rfm.index == int(cid)
            ]

            if customer.empty:

                st.error(
                    "Customer ID not found."
                )

            else:

                st.success(
                    "Customer Found"
                )

                st.dataframe(
                    customer
                )

                c1, c2, c3 = st.columns(3)

                c1.metric(

                    "Health Score",

                    round(
                        customer[
                            "Health_Score"
                        ].iloc[0],
                        2
                    )

                )

                c2.metric(

                    "CLV",

                    round(
                        customer[
                            "CLV"
                        ].iloc[0],
                        2
                    )

                )

                c3.metric(

                    "Churn Risk",

                    str(
                        round(
                            customer[
                                "Churn_Prob"
                            ].iloc[0] * 100,
                            2
                        )
                    ) + "%"

                )

                st.write(
                    "### Segment"
                )

                st.info(
                    customer[
                        "Segment"
                    ].iloc[0]
                )

                st.write(
                    "### Recommended Business Action"
                )

                st.success(
                    customer[
                        "Business_Action"
                    ].iloc[0]
                )

        except:

            st.error(
                "Please enter a valid Customer ID."
            )

# ============================================================
# SEGMENTATION
# ============================================================

elif menu == "Segmentation":

    st.header(
        "👥 Customer Segmentation"
    )

    fig = px.histogram(

        rfm,

        x="Segment",

        color="Segment",

        title="Customer Segments"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        rfm.head(20)
    )

# ============================================================
# CHURN ANALYSIS
# ============================================================

elif menu == "Churn Analysis":

    st.header(
        "⚠️ High Churn Risk Customers"
    )

    churn = rfm.sort_values(

        "Churn_Prob",

        ascending=False

    ).head(20)

    st.dataframe(
        churn
    )

# ============================================================
# CLV ANALYSIS
# ============================================================

elif menu == "CLV Analysis":

    st.header(
        "💰 Customer Lifetime Value"
    )

    top = rfm.sort_values(

        "CLV",

        ascending=False

    ).head(10)

    fig = px.bar(

        top,

        x=top.index.astype(str),

        y="CLV",

        title="Top Customers by CLV"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        top
    )

# ============================================================
# BUSINESS ACTIONS
# ============================================================

elif menu == "Business Actions":

    st.header(
        "🎯 Business Action Recommendations"
    )

    st.dataframe(

        rfm[
            [
                "Recency",
                "Frequency",
                "Monetary",
                "CLV",
                "Health_Score",
                "Churn_Prob",
                "Business_Action"
            ]
        ].sort_values(

            "Churn_Prob",

            ascending=False

        )

    )

    # ============================================================
# TOP CUSTOMERS
# ============================================================

elif menu == "Top Customers":

    st.header(
        "⭐ Top 10 VIP Customers"
    )

    vip = rfm.sort_values(

        "CLV",

        ascending=False

    ).head(10)

    fig = px.bar(

        vip,

        x=vip.index.astype(str),

        y="CLV",

        color="CLV",

        title="Top Customers by Lifetime Value"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        vip
    )

# ============================================================
# HIGH RISK CUSTOMERS
# ============================================================

elif menu == "High Risk Customers":

    st.header(
        "🚨 High Risk Customers"
    )

    risk = rfm[

        rfm["Churn_Prob"] > 0.70

    ].sort_values(

        "Churn_Prob",

        ascending=False

    )

    st.dataframe(
        risk
    )

# ============================================================
# MONTHLY SALES
# ============================================================

elif menu == "Monthly Sales":

    st.header(
        "📅 Monthly Revenue Analysis"
    )

    if "Month" in df.columns:

        monthly = (

            df.groupby(
                "Month"
            )["TotalAmount"]

            .sum()

            .reset_index()

        )

        fig = px.line(

            monthly,

            x="Month",

            y="TotalAmount",

            markers=True,

            title="Monthly Revenue Trend"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.dataframe(
            monthly
        )

    else:

        st.warning(
            "Month column not found."
        )

# ============================================================
# COUNTRY ANALYTICS
# ============================================================

elif menu == "Country Analytics":

    st.header(
        "🌍 Country Revenue Analysis"
    )

    country = (

        df.groupby(
            "Country"
        )["TotalAmount"]

        .sum()

        .reset_index()

        .sort_values(

            "TotalAmount",

            ascending=False

        )

        .head(15)

    )

    fig = px.bar(

        country,

        x="Country",

        y="TotalAmount",

        color="TotalAmount",

        title="Top Revenue Generating Countries"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        country
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif menu == "Feature Importance":

    st.header(
        "🧠 Machine Learning Feature Importance"
    )

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    fig = px.bar(

        importance,

        x="Feature",

        y="Importance",

        color="Importance",

        title="Random Forest Feature Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        importance
    )

# ============================================================
# TOP PRODUCTS
# ============================================================

elif menu == "Top Products":

    st.header(
        "🛒 Top Selling Products"
    )

    products = (

        df.groupby(
            "Description"
        )["Quantity"]

        .sum()

        .reset_index()

        .sort_values(

            "Quantity",

            ascending=False

        )

        .head(20)

    )

    fig = px.bar(

        products,

        x="Description",

        y="Quantity",

        color="Quantity",

        title="Most Purchased Products"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(
        products
    )


# ============================================================
# CUSTOMER BATTLE ARENA
# ============================================================

elif menu == "Customer Battle Arena":

    st.header("⚔️ Customer Battle Arena")

    customer_ids = sorted(rfm.index.astype(int).tolist())

    col1, col2 = st.columns(2)

    with col1:
        customer1 = st.selectbox("Customer A", customer_ids)

    with col2:
        customer2 = st.selectbox("Customer B", customer_ids, index=1)

    if customer1 != customer2:

        A = rfm.loc[customer1]
        B = rfm.loc[customer2]

        compare = pd.DataFrame({
            "Metric":["Segment","Health Score","CLV","Churn Risk %","Recency","Frequency","Monetary"],
            "Customer A":[A["Segment"],round(A["Health_Score"],2),round(A["CLV"],2),round(A["Churn_Prob"]*100,2),A["Recency"],A["Frequency"],round(A["Monetary"],2)],
            "Customer B":[B["Segment"],round(B["Health_Score"],2),round(B["CLV"],2),round(B["Churn_Prob"]*100,2),B["Recency"],B["Frequency"],round(B["Monetary"],2)]
        })

        st.dataframe(compare, use_container_width=True)

        scoreA = A["Health_Score"] + A["CLV"]/100 - A["Churn_Prob"]*50
        scoreB = B["Health_Score"] + B["CLV"]/100 - B["Churn_Prob"]*50

        st.markdown("---")

        if scoreA > scoreB:
            st.success(f"🏆 Winner : Customer {customer1}")
        else:
            st.success(f"🏆 Winner : Customer {customer2}")

    else:
        st.warning("Please select two different customers.")


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "© 2026 Customer Intelligence Platform | Built with Python • Machine Learning • Streamlit"
)
