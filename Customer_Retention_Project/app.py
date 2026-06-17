import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🚀",
    layout="wide"
)

# ---------------- SESSION ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

.big-title{
    text-align:center;
    font-size:70px;
    font-weight:bold;
    color:#1f4e79;
}

.subtitle{
    text-align:center;
    font-size:50px;
    color:gray;
}

.feature-box{
    background-color:;black;
    color:white;
    padding:20px;
    border-radius:15px;
    box-shadow:2px 2px 15px rgba(0,0,0,0.1);
    margin-bottom:15px;
}

.footer{
    text-align:center;
    color:black;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LANDING PAGE ----------------

if st.session_state.logged_in == False:

    st.markdown(
        '<p class="big-title">🚀 Customer Intelligence Platform</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">AI Powered Customer Segmentation & Retention Analytics</p>',
        unsafe_allow_html=True
    )

    st.write("")

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        use_container_width=True
    )

    st.write("")
    st.write("")

    st.header("📌 About The Platform")

    st.write("""
This enterprise dashboard helps organizations understand customer behaviour,
predict churn, estimate customer lifetime value and automatically recommend
business actions using Machine Learning.
""")

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">

        ## 👥 Customer Segmentation

        - RFM Analysis
        - Active Customers
        - Inactive Customers

        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">

        ## ⚠️ Churn Prediction

        - Machine Learning
        - Risk Detection
        - Retention Planning

        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">

        ## 💰 Customer Value

        - CLV Estimation
        - VIP Customers
        - Marketing Insights

        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.header("📈 Platform Features")

    st.write("""
✅ Executive Dashboard

✅ Customer Search

✅ Customer Segmentation

✅ Churn Prediction

✅ CLV Analysis

✅ Business Action Recommendations

✅ VIP Customer Identification

✅ High Risk Customer Detection

✅ Country Analytics

✅ Monthly Revenue Analysis

✅ Downloadable Reports

""")

    st.write("")
    st.write("")

    st.header("🔐 Login")

    username = st.text_input(
        "Username"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if (
            username == "admin"
            and
            password == "admin"
        ):

            st.session_state.logged_in = True

            st.success(
                "Login Successful!"
            )

            st.switch_page(
                "pages/dashboard.py"
            )

        else:

            st.error(
                "Invalid Username or Password"
            )

    st.markdown("""
    <div class="footer">

    Built using Python • Machine Learning • Streamlit

    </div>
    """, unsafe_allow_html=True)

# ---------------- REDIRECT ----------------

else:

    st.switch_page(
        "pages/dashboard.py"
    )