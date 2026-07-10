from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Prediction History",
    page_icon="📜",
    layout="wide"
)

st.title("📜 Prediction History")

# =====================================================
# DATABASE PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "database" / "customer.db"

# =====================================================
# CONNECT DATABASE
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

# =====================================================
# LOAD HISTORY
# =====================================================

history = pd.read_sql_query(
    "SELECT * FROM prediction_history ORDER BY id DESC",
    conn
)
st.write(history)

# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total prediction_history",
    len(history)
)

if len(history) > 0:

    high = len(history[history["risk"] == "High"])
    medium = len(history[history["risk"] == "Medium"])
    low = len(history[history["risk"] == "Low"])

else:

    high = medium = low = 0

col2.metric(
    "High Risk",
    high
)

col3.metric(
    "Low Risk",
    low
)

st.markdown("---")

# =====================================================
# HISTORY TABLE
# =====================================================

st.subheader("Prediction Records")

if history.empty:

    st.info("No prediction history available.")

else:

    st.dataframe(
        history,
        use_container_width=True
    )

# =====================================================
# DOWNLOAD
# =====================================================

if not history.empty:

    st.download_button(
        label="📥 Download Prediction History",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

conn.close()
