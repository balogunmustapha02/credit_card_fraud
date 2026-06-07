# 🔍 Credit Card Fraud Detector

A machine learning web application that detects fraudulent credit card transactions using four trained models. Built with Python and deployed via Streamlit.

---

## 🚀 Live Demo

[**View App →**](https://your-username-credit-card-fraud-detector.streamlit.app)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Deployment](#deployment)

---

## Overview

This project trains and deploys four machine learning models on an imbalanced credit card transaction dataset to classify transactions as **fraudulent** or **legitimate**. The app provides both single transaction prediction (manual entry) and batch prediction (CSV upload), with a live model comparison table when true labels are available.

---

## Models

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline linear classifier |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting — typically strongest on tabular data |
| **Neural Network** | Deep learning model built with Keras/TensorFlow |

All models were trained with class imbalance handling due to the heavily skewed fraud-to-legitimate ratio (~0.17% fraud).

---

## Features

- 🔄 **Switch between all 4 models** from the sidebar
- 🎚️ **Adjustable decision threshold** — tune sensitivity to fraud
- 📋 **Manual entry** — input a single transaction's features and predict
- 📂 **CSV batch upload** — predict on multiple transactions at once
- 📊 **Model comparison table** — benchmark all 4 models side by side (AUC-ROC, F1, Precision, Recall) when true labels are provided
- 📈 **Probability distribution chart** — visualize fraud vs legitimate score separation
- ⬇️ **Download results** — export predictions as CSV

---

## Project Structure

```
credit-card-fraud-detector/
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
└── saved_models/
    ├── lr_model.pkl        # Logistic Regression
    ├── rf_model.pkl        # Random Forest
    ├── xgb_model.json      # XGBoost (native format)
    ├── nn_model.keras      # Neural Network
    ├── scaler_amount.pkl   # StandardScaler for Amount
    └── scaler_time.pkl     # StandardScaler for Time
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/balogunmustapha02/credit_card_fraud.git
cd credit-card-fraud
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

### Manual Entry
1. Select a model from the sidebar
2. Adjust the decision threshold if needed
3. Enter the transaction features (Time, Amount, V1–V28)
4. Click **Predict**

### Batch Prediction (CSV)
1. Switch to **Upload CSV** mode
2. Upload a CSV with columns: `Time, V1–V28, Amount` (optional: `Class`)
3. View results, fraud distribution chart, and model comparison table
4. Download predictions as CSV

---

## Dataset

The model was trained on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **284,807** transactions
- **492** fraudulent (0.17%)
- Features V1–V28 are PCA-transformed for confidentiality
- `Time` and `Amount` are the only original features

---

## Results

                             Accuracy  Precision  Recall      F1  ROC-AUC
Model                                                                    
Random Forest (Tuned)          0.9995     0.8824  0.7895  0.8333   0.9436
XGBoost (Tuned)                0.9995     0.9714  0.7158  0.8242   0.9752
Neural Network                 0.9993     0.7912  0.7579  0.7742   0.9323
Logistic Regression (Tuned)    0.9989     0.6466  0.7895  0.7109   0.9619

---

## Deployment

Deployed on **Streamlit Community Cloud**.

To deploy your own instance:
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set main file to `app.py` and click **Deploy**

---

## Dependencies

```
pandas==3.0.3
numpy==2.4.6
tensorflow==2.21.0
xgboost==3.2.0
joblib==1.5.3
streamlit==1.58.0
matplotlib==3.10.9
seaborn==0.13.2
dill==0.4.1
scikit-learn==1.9.0
```

---

## Author

**Group 22 C6** — Techcrush C6 AI/ML Track - Class A
[GitHub](https://github.com/YOUR_USERNAME)
