import sqlite3
import pandas as pd

conn = sqlite3.connect("database/customer.db")

df = pd.read_sql_query(
    "SELECT * FROM prediction_history",
    conn
)

print(df)

conn.close()
