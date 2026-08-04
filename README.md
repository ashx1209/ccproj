# 💳 Credit Card Default Prediction & Risk Assessment

## 📌 Project Overview
Financial institutions face significant risk and revenue loss from credit card client defaults. The goal of this project is to build a robust Machine Learning pipeline that predicts the probability of a client defaulting on their credit card payment in the upcoming month. 

This repository contains an end-to-end data science project: from exploratory data analysis and handling severe class imbalance, to model tuning, secure artifact packaging, and a dynamic front-end deployment using **Streamlit**.

---

## 🏗️ Project Architecture & Methodology

1. **Data Preprocessing & Cleaning:** Standardized categorical variables (e.g., grouping unknown education/marital statuses) and formatted financial history features.
2. **Handling Class Imbalance:** Evaluated multiple techniques including Class Weight balancing and **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model learns default patterns without bias toward the majority class.
3. **Feature Scaling & Engineering:** Built a robust column-alignment pipeline using Pandas `.reindex()` to ensure production data perfectly matches training architecture, alongside a `StandardScaler` for distance-based algorithms.
4. **Model Selection & Tuning:** Trained and evaluated Baseline Random Forest, Balanced Random Forest, and RBF Support Vector Machines (SVM).

---

## 📊 Key Results & Model Selection

The models were evaluated on an **unseen test set** using Precision, Recall, and the F1-Score (prioritizing the minority "Default" class).

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Tuned RBF SVM (Champion)** | **77.55%** | **49.33%** | **55.84%** | **0.5239** |
| Balanced RF (30% Threshold) | 78.48% | 51.31% | 53.05% | 0.5217 |
| SMOTE RF (30% Threshold) | 64.25% | 35.13% | 72.80% | 0.4739 |
| Baseline Random Forest | 81.52% | 64.49% | 36.55% | 0.4666 |

**Business Decision:** The **Tuned RBF SVM** was selected as the final production model. It strikes the optimal mathematical balance (highest overall F1-score) between catching actual defaults (~56%) while maintaining a clean precision rate (~50%), minimizing unnecessary customer friction from false alarms.

---

## 💻 The Web Application
The final model is deployed via a dynamic **Streamlit** web application. 

**Features:**
* **Dynamic Financial History:** Users can toggle between inputting 1 to 6 months of historical billing and payment data using Session State.
* **Production-Safe Preprocessing:** The app automatically handles missing inputs by defaulting unknown financial months to zero and perfectly aligning one-hot encoded columns using the exported `expected_columns.pkl`.
* **Instant Inference:** Runs inputs through the saved standard scaler and SVM model to provide immediate risk assessment.

---

## 📂 Repository Structure

```text
credit-card-default-prediction/
│
├── data/                       # Directory for the dataset (ignored in git)
│
├── notebooks/                  
│   └── model_training.ipynb    # EDA, SMOTE, Model Training, and Evaluation
│
├── models/                     # Exported ML artifacts for production
│   ├── expected_columns.pkl    # Extracted training column names for alignment
│   ├── feature_scaler.pkl      # Fitted StandardScaler
│   └── final_tuned_svm_model.pkl # The final trained RBF SVM
│
├── app.py                      # Streamlit application script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation