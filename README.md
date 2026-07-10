# 📊 Customer Churn Prediction using Machine Learning

An end-to-end Machine Learning project that predicts customer churn using the IBM Telco Customer Churn dataset. The project includes data preprocessing, exploratory data analysis (EDA), model training, prediction, business insights, and an interactive Streamlit dashboard.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for telecom companies. This project predicts whether a customer is likely to leave the service based on customer demographics, subscription details, and billing information.

The solution helps businesses identify high-risk customers and take proactive retention actions.

---

## 🚀 Features

- 📊 Interactive Streamlit Dashboard
- 🤖 Customer Churn Prediction
- 📈 Model Performance Evaluation
- 📜 Prediction History (SQLite Database)
- 📊 Feature Importance Visualization
- 💼 Business Insights
- 📋 Customer Information Viewer

---

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn

- Total Records: **7043**
- Features: **20**
- Target Variable: **Churn**
- Missing Values handled during preprocessing

Target values:

- Yes → Customer Churned
- No → Customer Stayed

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary columns (`customerID`)
- Converted `TotalCharges` to numeric
- Handled missing values using median imputation
- Label Encoding for categorical features
- Feature Scaling using StandardScaler
- Train-Test Split (80:20)

---

## 📊 Exploratory Data Analysis (EDA)

EDA includes:

- Dataset Overview
- Missing Value Analysis
- Churn Distribution
- Contract Type Analysis
- Internet Service Analysis
- Payment Method Analysis
- Monthly Charges Distribution
- Tenure Distribution
- Correlation Heatmap
- Feature Importance

---

## 🤖 Machine Learning Model

**Algorithm Used**

- Random Forest Classifier

Why Random Forest?

- Handles categorical data efficiently
- Robust against overfitting
- High predictive performance
- Provides Feature Importance

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **79.63%** |
| Precision | **66.17%** |
| Recall | **47.18%** |
| F1 Score | **55.09%** |

Confusion Matrix

```
[[946  90]
 [197 176]]
```

---

## 💡 Business Insights

Key findings from the dataset:

- Month-to-Month contract customers have the highest churn risk.
- Fiber Optic customers show higher churn compared to other internet services.
- Customers with higher monthly charges are more likely to churn.
- Long-tenure customers are generally more loyal.
- Long-term contracts significantly reduce churn.

---

## 🏗️ Project Architecture

```
IBM Telco Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Random Forest Model
        │
        ▼
Model Evaluation
        │
        ▼
Saved Model (model.pkl)
        │
        ▼
Streamlit Dashboard
        │
        ├── Dashboard
        ├── Churn Prediction
        ├── Prediction History
        └── Model Performance
```

---

## 📁 Project Structure

```
Customer_Retention_Project/

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
├── database/
│   └── customer.db
│
├── ml/
│   ├── train_model.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── metrics.pkl
│
├── eda.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sanikaashinde/Customer-Retention-project.git
```

Go to the project folder

```bash
cd Customer-Retention-project
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
python database/init_db.py
streamlit run dashboard/app.py
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- SQLite
- Joblib

---

## 🎯 Future Improvements

- XGBoost and LightGBM comparison
- Hyperparameter Tuning
- SHAP Explainability
- Customer Retention Recommendation Engine
- Model Deployment using Docker
- REST API Integration

---
## 📸 Dashboard Screenshots

### Home Page
<img width="1339" height="608" alt="app 1" src="https://github.com/user-attachments/assets/b7dcf4ef-1311-4d9e-aea8-db7e75a20ed4" />
<img width="1337" height="584" alt="app 2" src="https://github.com/user-attachments/assets/7281e051-e45b-423a-ab5c-4c9d5f61b5e5" />
<img width="1264" height="553" alt="app 3" src="https://github.com/user-attachments/assets/9f43da2a-91b7-4b5a-ba96-1615ac1db2db" />

### Dashboard
<img width="1326" height="643" alt="dashboard 1" src="https://github.com/user-attachments/assets/22cf8ee5-0890-4d84-b2e6-47dc7c8468b2" />
<img width="1340" height="635" alt="dashboard 2" src="https://github.com/user-attachments/assets/8e58b48f-f9c5-4b1d-89be-6dde4e05df3e" />
<img width="1332" height="629" alt="dashboard 3" src="https://github.com/user-attachments/assets/13deec0b-49e6-4ea5-b669-fb0d481d0c83" />
<img width="1336" height="609" alt="dashboard 4" src="https://github.com/user-attachments/assets/d665f0ae-de0d-49f8-b004-a0f6cdb7379a" />
<img width="1286" height="563" alt="dashboard 5" src="https://github.com/user-attachments/assets/d996a092-1714-474e-9d45-68164fb2b4ab" />

### Churn Prediction
<img width="1320" height="596" alt="chur predictio 1" src="https://github.com/user-attachments/assets/da529043-b100-4192-af71-9e5d2c85e679" />
<img width="1284" height="559" alt="chur predictio 2" src="https://github.com/user-attachments/assets/b6b62770-b4b2-4c21-985d-dea0939813d8" />

### Model Performance
<img width="1328" height="585" alt="model performance 1" src="https://github.com/user-attachments/assets/a5cb1029-32e8-4c1b-94d1-dc121d585b6e" />
<img width="1286" height="601" alt="model performance 2" src="https://github.com/user-attachments/assets/f1ae0e9e-19e4-4310-a569-4d4cc202c98c" />
<img width="1296" height="620" alt="model performance 3" src="https://github.com/user-attachments/assets/6b7e468d-e2f0-4e41-a805-c1c341897bb4" />

### Prediction History
<img width="1334" height="641" alt="predictio history" src="https://github.com/user-attachments/assets/729b1b1c-575f-4ff0-9c19-ccfc61a23a6f" />

## 👩‍💻 Author

**Sanika Shinde**

GitHub: https://github.com/sanikaashinde

---


