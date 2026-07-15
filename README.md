# Customer Churn Prediction System

A Machine Learning-based **Customer Churn Prediction System** built using **Python, Scikit-learn, Streamlit, SQLite, Pandas, and Joblib**.

This project predicts whether a telecom customer is likely to **churn (leave the service)** based on customer demographics, service usage, billing information, and contract details. It also provides an interactive Streamlit dashboard for customer predictions, model evaluation, business insights, and prediction history.

---

# Features

* Customer Churn Prediction using Machine Learning
* End-to-End Data Preprocessing Pipeline
* Missing Value Handling
* Duplicate Record Removal
* Categorical Feature Encoding
* Feature Scaling using StandardScaler
* Random Forest Classification Model
* Dynamic Model Performance Dashboard
* Prediction History using SQLite
* Interactive Streamlit Dashboard
* Reproducible Training Pipeline
* Saved Model Artifacts for Deployment

---

# Project Structure

```text
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
│   ├── churn.csv
│   ├── train.csv
│   └── test.csv
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

**Dataset Used**

IBM Telco Customer Churn Dataset

The dataset contains customer demographic, subscription, billing, and service information, including:

* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Multiple Lines
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Paperless Billing
* Payment Method
* Monthly Charges
* Total Charges
* Churn (Target Variable)

---

# Machine Learning Pipeline

The complete ML pipeline performs:

* Data Loading
* Duplicate Removal
* Missing Value Handling
* Numerical Feature Scaling
* Categorical Feature Encoding
* Train-Test Split (80:20)
* Random Forest Model Training
* Model Evaluation
* Model Artifact Saving

Generated artifacts:

* `model.pkl`
* `scaler.pkl`
* `encoders.pkl`
* `metrics.pkl`

---

# Model

**Algorithm**

Random Forest Classifier

**Random State**

```text
42
```

**Train-Test Split**

```text
80 : 20
```

---

# Model Performance

The latest trained Random Forest model achieved the following performance on the test dataset.

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 75.73% |
| Precision | 52.95% |
| Recall    | 76.74% |
| F1 Score  | 62.66% |
| ROC AUC   | 84.25% |

These metrics are automatically generated during model training and saved in:

```text
ml/metrics.pkl
```

The Streamlit dashboard loads these metrics dynamically from `ml/metrics.pkl`, ensuring that the displayed values always match the latest trained model.

---

# Database

SQLite is used to store prediction history.

**Database**

```text
database/customer.db
```

**Prediction Table**

```text
prediction_history
```

Stored columns:

* id
* customerID
* prediction
* probability
* risk
* created_at

Recreate the database anytime using:

```bash
python database/init_db.py
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sanikaashinde/Customer-Retention-project.git
```

Navigate to the project directory:

```bash
cd Customer_Retention_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Train the Model

Run:

```bash
python ml/train_model.py
```

The training pipeline automatically:

* Loads the dataset
* Removes duplicate records
* Handles missing values
* Encodes categorical features
* Splits the dataset into training and testing sets
* Saves `train.csv` and `test.csv`
* Applies feature scaling
* Trains the Random Forest model
* Evaluates model performance
* Saves model artifacts

Generated files:

```text
data/train.csv
data/test.csv

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

This recreates the `prediction_history` table if required.

---

# Run Dashboard

Launch the Streamlit application:

```bash
streamlit run dashboard/app.py
```

Dashboard modules:

* Dashboard Overview
* Churn Prediction
* Prediction History
* Model Performance

---

# Smoke Test

Verify that the trained model and preprocessing artifacts load successfully:

```bash
python test_smoke.py
```

---

# Screenshots

Application screenshots are available in the `screenshots/` folder.

Included screenshots:

* Application Home
* Dashboard Overview
* Churn Prediction
* Prediction Result
* Prediction History
* Model Performance

---

# Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* SQLite
* Joblib
* Plotly

---

# Requirements

Recommended package versions:

```text
streamlit==1.49.0
pandas==2.3.1
numpy==2.3.1
scikit-learn==1.7.1
joblib==1.5.1
plotly==6.2.0
sqlalchemy==2.0.41
matplotlib==3.10.3
```

---

# Project Highlights

* End-to-End Machine Learning Pipeline
* Reproducible Model Training
* Automatic Train/Test Dataset Generation
* Dynamic Model Performance Loading
* Modular Project Structure
* Interactive Streamlit Dashboard
* SQLite Integration
* Prediction History Tracking
* Deployment Ready

---

# Recent Improvements

Major updates include:

* Replaced the previous RFM-based implementation with a complete Customer Churn Prediction pipeline.
* Added an end-to-end reproducible `train_model.py` training script.
* Automatically generates `train.csv` and `test.csv` during training.
* Saves preprocessing artifacts (`model.pkl`, `scaler.pkl`, `encoders.pkl`, and `metrics.pkl`).
* Dashboard now loads evaluation metrics dynamically from `metrics.pkl`.
* Rebuilt the SQLite database with the `prediction_history` table.
* Removed obsolete files, duplicate resources, and merge-conflict artifacts.
* Updated the Streamlit dashboard to use the latest trained model and database schema.

---

# Author

**Sanika Shinde**

GitHub:

https://github.com/sanikaashinde
