import sqlite3

conn = sqlite3.connect("customer.db")

cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS predictions")

cursor.execute("""
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id REAL,
    recency REAL,
    frequency REAL,
    monetary REAL,
    probability REAL,
    risk_level TEXT,
    prediction TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

print("Table recreated successfully!")