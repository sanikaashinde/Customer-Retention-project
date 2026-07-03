import streamlit as st
import sqlite3
import pandas as pd

st.title("📜 Prediction History")

conn = sqlite3.connect(
    "../database/customer.db"
)

df = pd.read_sql(
    "SELECT * FROM predictions ORDER BY id DESC",
    conn
)

conn.close()

if df.empty:
    st.info("No predictions available.")

else:
    st.dataframe(
        df,
        use_container_width=True
    )