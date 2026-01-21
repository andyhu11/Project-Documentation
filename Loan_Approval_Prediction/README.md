# Loan Approval Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20SHAP-orange)
![Optimization](https://img.shields.io/badge/Tuning-RandomizedSearchCV-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A machine learning pipeline designed to automate and optimize the loan eligibility assessment process with high precision and fairness.**

## 📖 Overview

**Loan Approval Prediction** is a predictive modeling project built with **Python**, designed to streamline financial decision-making. By leveraging historical applicant data, this project replaces manual underwriting heuristics with data-driven classification.

The system manages the end-to-end machine learning lifecycle—from rigorous **Exploratory Data Analysis (EDA)** to **SHAP-based model interpretation**. It prioritizes high precision in the automatic-approval band to minimize risk while ensuring regulatory compliance through transparent feature impact analysis.

---

## ✨ Key Features

### 🛠 Core Engineering
* **Robust Preprocessing:** * Automated pipelines for **Winsorization** (outlier clipping) and **Log Transformations** to stabilize skewed financial distributions.
    * **Median Imputation** strategies for handling missing values in engineered features.
* **Advanced Feature Engineering:** * **Financial Ratios:** Construction of high-impact interaction features such as `Debt-to-Income Ratio`, `Total Assets` (Residential + Commercial + Luxury + Bank), and `Loan Amount per Term`.
    * **Categorical Encoding:** Standardized cleaning and encoding of complex categorical variables.

### 🧠 Model Architecture & Optimization
* **Multi-Model Evaluation:** Benchmarked **Logistic Regression**, **Support Vector Machines (SVM)**, and **Random Forest** classifiers.
* **Hyperparameter Tuning:** Utilized **RandomizedSearchCV** with Stratified K-Fold cross-validation to optimize decision boundaries.
* **Threshold Tuning:** Implemented precision-recall curve analysis to select decision thresholds that maximize recall while maintaining **>99% precision** for the positive class.

### 🔍 Explainability & Governance
* **SHAP Analysis:** Integration of **SHAP (SHapley Additive exPlanations)** to provide local and global explanations for individual loan decisions.
* **Feature Importance:** Global ranking of key drivers (e.g., *CIBIL Score*, *Loan Term*, *Asset Value*) to ensure alignment with financial intuition.

---

## 📈 Performance

Empirical analysis on the test set demonstrates that the **Random Forest** model significantly outperforms linear baselines, achieving near-perfect classification on this dataset.

### Model Comparison (Test Set)

| Model Architecture | Accuracy | ROC-AUC | Precision (Class 1) | Recall (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 92.15% | 0.9781 | 92.80% | 94.73% |
| **SVM (RBF Kernel)** | 95.55% | 0.9944 | 95.56% | 97.36% |
| **Random Forest (Tuned)**| **99.53%** | **0.9991** | **99.25%** | **100.0%** |

> **Key Insight:** The **Random Forest** model achieves a **100% recall** for the positive class while maintaining **99.25% precision**, making it the ideal candidate for an automated approval system where missing a viable customer (False Negative) is costly, but approving a defaulter (False Positive) is critical to avoid.

### Top Predictive Features
1. **CIBIL Score:** The dominant predictor for loan stability.
2. **Debt-to-Income Ratio:** A custom engineered feature reflecting repayment capacity.
3. **Loan Term:** Shorter terms correlate strongly with higher approval rates.

---

## 📂 Project Structure

```text
Loan_Approval_Prediction/
├── Loan Approval Prediction.ipynb          # End-to-end Machine Learning Workflow
└── README.md                               # Project Documentation

```

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.8+**
* **Jupyter Notebook** or **JupyterLab**
* **Key Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `joblib`

### Installation

1. Clone the repository:

```bash
git clone [https://github.com/your-username/Loan-Approval-Prediction.git](https://github.com/your-username/Loan-Approval-Prediction.git)

```

2. Navigate to the project directory:

```bash
cd Loan-Approval-Prediction

```

3. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib

```

### Usage Guide

1. **Launch:** Start the Jupyter environment.

```bash
jupyter notebook "Loan Approval Prediction.ipynb"

```

2. **Execution:** Run the cells sequentially to reproduce the analysis.
* **Step 1:** Data Loading & EDA (Distribution analysis, Outlier detection).
* **Step 2:** Preprocessing & Feature Engineering (Ratio creation).
* **Step 3:** Model Training & Tuning (Random Forest, SVM, LR).
* **Step 4:** Evaluation (ROC-AUC, Confusion Matrix, SHAP plots).



---

## 🚧 Roadmap & Future Enhancements

The following initiatives are planned to move the model from a strong baseline to a production-ready system:

* **Cost-Sensitive Optimization:**
* Refine the loss function to heavily penalize False Positives (bad loans approved) based on actual financial risk weights.


* **Production Deployment:**
* Wrap the model in a **FastAPI** service for real-time inference.
* Implement automated retraining triggers based on data drift detection.


* **Deep Compliance Suite:**
* Integrate **Fairness Metrics** (Demographic Parity, Equalized Odds) to automatically audit model decisions for bias against protected groups.



---

## 🤝 Contributing

Contributions to improve model performance or feature engineering are welcome.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewAlgorithm`).
3. Commit your Changes (`git commit -m 'Add Gradient Boosting experiment'`).
4. Push to the Branch (`git push origin feature/NewAlgorithm`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
