# Customer Churn Prediction System

A Machine Learning based Customer Churn Prediction System built using **Python, Scikit-learn, Streamlit, SQLite, and Pandas**.

The project predicts whether a telecom customer is likely to churn based on customer demographics, service usage, billing information, and contract details. It also provides an interactive Streamlit dashboard for predictions, model evaluation, and prediction history.

---

# Features

- Customer Churn Prediction using Machine Learning
- End-to-End Data Preprocessing Pipeline
- Missing Value Handling
- Categorical Feature Encoding
- Feature Scaling using StandardScaler
- Random Forest Classification Model
- Model Performance Dashboard
- Prediction History using SQLite
- Interactive Streamlit Dashboard
- Reproducible Training Pipeline
- Saved Model Artifacts for Deployment

---

# Project Structure

```
Customer_Retention_Project/
│
├── README.md
├── requirements.txt
├── test_smoke.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── dashboard.py
│       ├── churn_prediction.py
│       ├── prediction_history.py
│       └── Model_Performance.py
│
├── data/
│   └── churn.csv
│
├── ml/
│   ├── train_model.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── metrics.pkl
│
├── database/
│   ├── customer.db
│   ├── init_db.py
│   ├── connection.py
│   ├── models.py
│   ├── save_prediction.py
│   └── view_data.py
│
├── reports/
│
└── screenshots/
```

---

# Dataset

Dataset Used:

**Telco Customer Churn Dataset**

The dataset contains customer demographic, subscription, and billing information including:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Online Security
- Tech Support
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target Variable)

---

# Machine Learning Pipeline

The complete ML pipeline performs:

- Data Loading
- Duplicate Removal
- Missing Value Handling
- Numerical Feature Scaling
- Categorical Feature Encoding
- Train/Test Split
- Random Forest Model Training
- Model Evaluation
- Artifact Saving

Artifacts generated:

- model.pkl
- scaler.pkl
- encoders.pkl
- metrics.pkl

---

# Model

Algorithm Used

**Random Forest Classifier**

Random State:

```
42
```

Train-Test Split:

```
80 : 20
```

---

# Model Performance

Current model performance:

| Metric | Value |
|---------|---------|
| Accuracy | 79.63% |
| Precision | 66.17% |
| Recall | 47.18% |
| F1 Score | 55.09% |

These metrics are automatically generated during model training and stored in:

```
ml/metrics.pkl
```

---

# Database

SQLite database is used to store prediction history.

Database:

```
database/customer.db
```

Prediction table:

```
prediction_history
```

Stored fields include:

- id
- customerID
- prediction
- probability
- risk
- created_at

Database can be recreated anytime using:

```bash
python database/init_db.py
```

---

# Installation

Clone Repository

```bash
git clone https://github.com/sanikaashinde/Customer-Retention-project.git
```

Move into project

```bash
cd Customer_Retention_Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Train the Model

Run:

```bash
python ml/train_model.py
```

This automatically:

- Loads the dataset
- Cleans data
- Encodes categorical features
- Scales numerical features
- Trains the Random Forest model
- Evaluates performance
- Saves model artifacts

Generated files:

```
ml/model.pkl
ml/scaler.pkl
ml/encoders.pkl
ml/metrics.pkl
```

---

# Initialize Database

Run:

```bash
python database/init_db.py
```

This recreates the prediction history table if required.

---

# Run Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard contains:

- Dashboard Overview
- Churn Prediction
- Prediction History
- Model Performance

---

# Smoke Test

To verify that the saved model loads correctly:

```bash
python test_smoke.py
```

---

# Screenshots

Application screenshots are available inside the **screenshots/** folder.

Included screenshots:

- Application Home
- Dashboard
- Churn Prediction
- Prediction Results
- Prediction History
- Model Performance

---

# Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- SQLite
- Joblib
- Plotly

---

# Requirements

Important package versions:

```
scikit-learn==1.2.2
pandas==1.5.3
numpy==1.24.3
streamlit==1.23.1
joblib==1.2.0
plotly==5.15.0
sqlalchemy==1.4.49
```

---

# Project Highlights

- End-to-End Machine Learning Pipeline
- Reproducible Model Training
- Modular Project Structure
- Streamlit Dashboard
- SQLite Integration
- Prediction History Tracking
- Model Performance Visualization
- Deployment Ready

---

# Recent Improvements

The project was updated to improve reproducibility and maintainability.

Major updates include:

- Replaced old RFM-based implementation with a true Customer Churn Prediction pipeline.
- Added a reproducible `train_model.py` script for end-to-end model training.
- Saved preprocessing artifacts (`model.pkl`, `scaler.pkl`, `encoders.pkl`, and `metrics.pkl`).
- Rebuilt the SQLite database with a clean `prediction_history` table.
- Removed obsolete files, duplicate resources, and merge-conflict artifacts.
- Updated the Streamlit dashboard to use the latest trained model and database schema.

---

# Author

**Sanika Shinde**

GitHub:

https://github.com/sanikaashinde
