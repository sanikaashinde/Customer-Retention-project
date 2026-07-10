import sqlite3

conn = sqlite3.connect("customer.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recency REAL,
    frequency REAL,
    monetary REAL,
    probability REAL,
    risk_level TEXT
)
""")

conn.commit()

conn.close()
