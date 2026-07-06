# 🚀 Customer Retention & Churn Prediction using Machine Learning

## 📌 Project Overview

This project is an AI-powered Customer Intelligence Platform that predicts whether a customer is likely to churn based on customer purchasing behavior using Machine Learning.

The project includes:

- Customer Segmentation using RFM Analysis
- Customer Churn Prediction
- Customer Lifetime Value (CLV) Analysis
- Business Recommendations
- FastAPI Backend
- Streamlit Interactive Dashboard
- SQLite Database Integration

---

## 🎯 Objectives

- Identify customers who are likely to churn
- Segment customers based on RFM values
- Predict churn probability using Machine Learning
- Help businesses improve customer retention
- Provide actionable business recommendations

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- FastAPI
- SQLite
- Plotly
- Joblib

---

## 📂 Project Structure

```
Customer_Retention_Project/

│
├── backend/
│   ├── main.py
│   ├── predict.py
│
├── dashboard/
│   └── pages/
│       └── dashboard.py
│
├── data/
│   ├── clean_online_retail.csv
│   └── customer_rfm_segments.csv
│
├── database/
│   ├── customer.db
│   ├── create_db.py
│   └── view_data.py
│
├── ml/
│   ├── train_model.py
│   ├── model.pkl
│   └── scaler.pkl
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
└── reports/
```

---

# Dataset

Dataset Used:

Online Retail II Dataset

Features used for Churn Prediction:

- Recency
- Frequency
- Monetary

Target Variable:

Churn

- 1 → Customer Will Churn
- 0 → Customer Will Stay

---

# Data Preprocessing

✔ Missing Value Handling

✔ Feature Selection

✔ Train-Test Split

✔ Feature Scaling using StandardScaler

---

# Machine Learning Model

Model Used:

Random Forest Classifier

---

# Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Example Results

Accuracy : 100%

Precision : 100%

Recall : 100%

F1 Score : 100%

---

# Features

## Dashboard

- Executive Overview
- Customer Search
- Customer Segmentation
- Revenue Forecast
- Churn Analysis
- Churn Prediction
- CLV Analysis
- Business Actions
- Top Customers
- High Risk Customers
- Country Analytics
- Top Products
- Customer Battle Arena
- API Documentation

---

# FastAPI Endpoints

## GET /

Returns API status.

## POST /predict

Input

```json
{
  "customer_id":12345,
  "recency":20,
  "frequency":8,
  "monetary":500
}
```

Output

```json
{
  "prediction":"Stay",
  "churn_probability":1,
  "risk_level":"Low",
  "recommendation":"Healthy customer..."
}
```

---

# Database

SQLite Database stores:

- Customer ID
- Recency
- Frequency
- Monetary
- Churn Probability
- Risk Level
- Prediction
- Timestamp

---

# Installation

Clone Repository

```
git clone https://github.com/sanikaashinde/Customer-Retention-project.git
```

Install Dependencies

```
pip install -r requirements.txt
```

Run FastAPI

```
cd backend

uvicorn main:app --reload
```

Run Streamlit

```
cd dashboard

streamlit run app.py
```

---

# Future Enhancements

- XGBoost Model
- Email Notification
- Live Database
- Cloud Deployment
- Authentication
- Customer Recommendation Engine

---
![Uploading 2.JPG…]()
![Uploading 4.JPG…]()
![Uploading 5.JPG…]()
![Uploading 6.JPG…]()
![Uploading 7.JPG…]()
![Uploading 8.JPG…]()

---
# Author

**Sanika Shinde**

B.Sc. Data Science

Machine Learning Project

2026

---

# License

This project is developed for educational and academic purposes.
