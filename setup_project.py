import os
import shutil

# Create folders
folders = [
    "dashboard",
    "ml",
    "data",
    "backend",
    "database"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Move files if they exist
files_to_move = {
    "app.py": "dashboard/app.py",
    "clean_online_retail.csv": "data/clean_online_retail.csv",
    "customer_rfm_segments.csv": "data/customer_rfm_segments.csv",
    "model.pkl": "ml/model.pkl",
    "scaler.pkl": "ml/scaler.pkl"
}

for src, dst in files_to_move.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {src} → {dst}")
    else:
        print(f"Not found: {src}")

# Move pages folder
if os.path.exists("pages"):
    shutil.move("pages", "dashboard/pages")
    print("Moved: pages → dashboard/pages")

print("\nProject structure created successfully!")