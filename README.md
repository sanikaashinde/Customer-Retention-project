# Customer Intelligence Platform

## Customer Retention Analysis Using Data Science

### Project Overview

The Customer Intelligence Platform is a data-driven analytics solution designed to help businesses understand customer behavior, improve customer retention, and make informed strategic decisions.

This project analyzes customer transaction data and applies customer segmentation techniques to identify valuable customers, loyal customers, potential churn customers, and other customer groups. The platform provides meaningful insights through an interactive Streamlit dashboard, enabling businesses to improve customer engagement and retention strategies.

---

## Problem Statement

In today's competitive business environment, retaining existing customers is as important as acquiring new ones. Many organizations struggle to identify customer purchasing patterns and understand which customers are likely to become inactive.

Without proper customer intelligence, businesses may experience:

* Reduced customer retention
* Lower revenue generation
* Ineffective marketing campaigns
* Poor customer engagement
* Lack of data-driven decision making

This project addresses these challenges by analyzing customer transaction data and generating actionable customer insights.

---

## Objectives

* Analyze customer transaction behavior.
* Clean and preprocess customer data.
* Perform customer segmentation using RFM Analysis.
* Identify high-value and at-risk customers.
* Generate customer intelligence reports.
* Visualize business insights through dashboards.
* Support customer retention strategies.

---

## Key Features

### Data Preprocessing

* Missing value handling
* Duplicate record removal
* Data cleaning and transformation
* Data quality improvement

### Customer Segmentation

* Recency Analysis
* Frequency Analysis
* Monetary Analysis
* RFM Score Generation
* Customer Category Identification

### Business Intelligence Dashboard

* Interactive visualizations
* Customer distribution analysis
* Revenue insights
* Segment-wise performance analysis
* Retention-focused reporting

### Reporting

* Customer Intelligence Reports
* Segmentation Reports
* Business Performance Insights

---

## Technology Stack

| Category              | Technologies                |
| --------------------- | --------------------------- |
| Programming Language  | Python                      |
| Data Analysis         | Pandas, NumPy               |
| Machine Learning      | Scikit-Learn                |
| Visualization         | Plotly, Matplotlib, Seaborn |
| Dashboard Development | Streamlit                   |
| Data Processing       | CSV Files                   |

---

## Project Structure

```text
Customer-Retention-Project
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── pages/
│   └── dashboard.py
│
├── clean_online_retail.csv
├── customer_rfm_segments.csv
├── customer_segmentation_final.csv
├── final_customer_intelligence_report.csv
│
└── models/
```

---

## Methodology

### Step 1: Data Collection

Customer transaction data is collected and loaded into the system.

### Step 2: Data Preprocessing

The raw data is cleaned by handling missing values, removing duplicates, and preparing it for analysis.

### Step 3: RFM Analysis

Customers are evaluated based on:

* Recency (Recent purchases)
* Frequency (Purchase frequency)
* Monetary Value (Amount spent)

### Step 4: Customer Segmentation

Customers are grouped into meaningful categories based on their purchasing behavior.

### Step 5: Dashboard Generation

Interactive dashboards and visual reports are created to provide business insights.

### Step 6: Business Intelligence

The generated insights help businesses improve customer retention and decision-making.

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/sanikaashinde/Customer-Retention-project>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser and display the Customer Intelligence Dashboard.

---

## Dashboard Insights

The dashboard provides:

* Customer Segment Distribution
* RFM Analysis
* Revenue Contribution Analysis
* Customer Behavior Analysis
* Customer Retention Insights
* Interactive Business Reports

---

## Expected Outcomes

This project helps organizations:

* Understand customer purchasing patterns
* Identify valuable customers
* Detect potential customer churn
* Improve customer retention
* Increase customer satisfaction
* Support data-driven business strategies

---

## Future Enhancements

* Advanced Machine Learning Models
* Customer Churn Prediction
* Recommendation System
* Real-Time Analytics
* Automated Reporting
* Cloud Deployment Integration

---

## Conclusion

The Customer Intelligence Platform transforms raw customer transaction data into actionable business intelligence. By combining data analysis, customer segmentation, and interactive visualization, the platform enables organizations to better understand their customers, enhance retention strategies, and improve overall business performance.

---

## Author

**Sanika Shinde**
B.Sc. Data Science

---

## License

This project is developed for educational and learning purposes.
