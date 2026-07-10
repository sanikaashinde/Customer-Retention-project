from sqlalchemy import create_engine

DATABASE_URL = "sqlite:///customer.db"

engine = create_engine(DATABASE_URL)

print("Database Connected Successfully")
