# 🛡️ NetGuard AI

### Machine Learning Powered Network Intrusion Detection System

NetGuard AI is a machine learning based Network Intrusion Detection System (NIDS) that analyzes network traffic and classifies connections as **Normal** or **Attack**.

The project compares multiple machine learning approaches and provides an interactive Streamlit dashboard for network traffic analysis and intrusion detection.

---

## 🚀 Features

- 🛡️ Binary network intrusion detection
- 🤖 Logistic Regression baseline
- 🌲 Random Forest classifier
- 📊 Model performance comparison
- 📈 ROC-AUC analysis
- 🎯 Precision-Recall evaluation
- 🔲 Confusion matrix visualization
- 🔍 Feature importance analysis
- 🌐 Interactive intrusion detection
- 📁 Batch CSV detection
- 📊 Network traffic analytics
- 📥 Downloadable detection results
- 🎨 Cybersecurity-themed Streamlit dashboard

---

## 🧠 Machine Learning Workflow

```text
Network Traffic Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Binary Target Creation
          ↓
Train-Test Split
          ↓
Feature Preprocessing
          ↓
One-Hot Encoding
          ↓
Feature Scaling
          ↓
Machine Learning
     ┌───────────────┐
     │ Logistic      │
     │ Regression    │
     └───────────────┘
            +
     ┌───────────────┐
     │ Random Forest │
     └───────────────┘
          ↓
Model Evaluation
          ↓
Normal / Attack