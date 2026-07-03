import sqlite3

conn = sqlite3.connect("../database/customer.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(predictions)")
print(cursor.fetchall())

conn.close()