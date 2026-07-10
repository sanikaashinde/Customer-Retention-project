import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "customer.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS prediction_history")

cursor.execute("""
CREATE TABLE prediction_history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customerID TEXT,
    prediction INTEGER,
    probability REAL,
    risk TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
INSERT INTO prediction_history
(customerID,prediction,probability,risk)
VALUES
('7590-VHVEG',0,0.18,'Low')
""")

conn.commit()
conn.close()

print("Database initialized successfully.")
