import sqlite3

conn = sqlite3.connect("../database/customer.db")
cursor = conn.cursor()

cursor.execute("""
ALTER TABLE predictions
ADD COLUMN prediction TEXT
""")
cursor.execute("""
ALTER TABLE predictions
ADD COLUMN prediction TEXT
""")

conn.commit()
conn.close()

print("Column added successfully.")