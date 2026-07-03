import sqlite3

conn = sqlite3.connect("../database/customer.db")
cursor = conn.cursor()

cursor.execute("""
ALTER TABLE predictions
ADD COLUMN customer_id REAL
""")

conn.commit()
conn.close()

print("customer_id column added.")