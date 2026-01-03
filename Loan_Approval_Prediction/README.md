# Loan Approval Prediction (Binary Classification)

> An end-to-end **loan approval (Approved/Rejected)** binary classification project using **scikit-learn**, implemented as a single Jupyter Notebook.  
> This project contains only one file: `Loan Approval Prediction.ipynb`

---

## 1. Project Overview

This notebook walks through a complete machine learning workflow:

**Data download → EDA → Visualization → Leakage-free preprocessing → Feature engineering → Model selection & tuning → Interpretability (incl. SHAP) → Threshold policy analysis → Test evaluation → Saving artifacts**

- Dataset source: Kaggle (`architsharma01/loan-approval-prediction-dataset`)
- Data file: `loan_approval_dataset.csv`
- Target: `loan_status` (binary: Approved / Rejected)
- Outputs: trained models (`.joblib`), threshold policy (`.json`), and evaluation metrics (`.json`) saved to `./artifacts/`

---

## 2. Features

### 2.1 Data Download & Loading
- Automatically downloads the Kaggle dataset using `kagglehub`
- Loads `loan_approval_dataset.csv` into a pandas DataFrame

### 2.2 Data Overview & Quality Checks
- Basic inspection: shape, dtypes, descriptive stats
- Missing value and duplicate checks
- Light “safe cleaning” that does not introduce leakage

### 2.3 Exploratory Data Analysis (EDA) & Visualization
The notebook includes a solid EDA section with plots such as:
- Numeric feature histograms (with KDE) and box plots (distribution & outliers)
- Categorical feature count plots
- Distribution comparisons after `log1p` transforms for skewed numeric variables
- Target distribution and target-vs-feature comparisons (KDE)
- Crosstabs of categorical features vs. the target (counts and class proportions)

### 2.4 Leakage-Free Preprocessing
Key principle: **split first, fit transformers only on the training set**.
- Normalizes the target values (lowercase/strip) and maps Approved/Rejected → 1/0
- Performs `train_test_split` with `stratify`
- Numeric features:
  - `SimpleImputer(strategy="median")` (fit on train, transform test)
  - `StandardScaler()` (fit on train, transform test)
- Categorical features:
  - One-hot encoding based on training schema, then aligned to test columns

### 2.5 Feature Engineering
Adds domain-inspired engineered features and merges them into the training matrix (still leakage-safe):
- `feat_debt_to_income`: `loan_amount / income_annum`
- `feat_total_assets`: sum of asset-related columns (wealth/collateral proxy)
- `feat_amt_per_term`: `loan_amount / loan_term` (approx. installment pressure)

Engineered features also go through:
- Median imputation (fit on train)
- Standard scaling (fit on train)
- Concatenation with the original feature matrix

### 2.6 Model Training & Hyperparameter Search (Cross-Validation)
Trains and tunes multiple models:
- Logistic Regression (baseline)
  - Searches `C` (log space)
  - `class_weight`: `None` / `balanced`
- Random Forest (strong baseline)
  - Random search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `class_weight`, etc.
- SVM (RBF)
  - Random search over `C`, `gamma`, etc. with `probability=True`

Uses:
- `RandomizedSearchCV` + `StratifiedKFold(5)`
- Primary selection metric: **ROC-AUC** (`scoring="roc_auc"`)

### 2.7 Model Interpretability
- Logistic Regression: coefficient analysis (direction & magnitude)
- Random Forest: feature importance
- SHAP:
  - Uses `shap.KernelExplainer` for model explanation
  - Generates SHAP summary plots for global feature impact

### 2.8 Threshold Policy & Decision Strategy
- Scans thresholds for Random Forest on the test set and plots how precision/recall/F1 vary
- Demonstrates two example operating points:
  - `t_cautious = 0.95`: more conservative (typically higher precision, lower recall)
  - `t_balanced = 0.80`: more balanced (typically higher recall)
- Outputs confusion matrices and key counts (TP/FP/FN/TN) per chosen threshold

### 2.9 Evaluation & Model Comparison
Compares LR / RF / SVM under default threshold 0.5:
- Accuracy, ROC-AUC, PR-AUC
- Precision/Recall/F1 for both positive and negative classes
- Confusion matrices
- Precision–Recall curves (including the baseline positive rate)

### 2.10 Saving Artifacts
At the end, the notebook creates `./artifacts/` and saves:
- Models:
  - `rf_best_YYYYMMDD-HHMMSS.joblib`
  - `lr_best_YYYYMMDD-HHMMSS.joblib`
  - `svm_best_YYYYMMDD-HHMMSS.joblib`
- Threshold policy:
  - `thresholds_YYYYMMDD-HHMMSS.json`
- Aggregated evaluation metrics:
  - `metrics_YYYYMMDD-HHMMSS.json`

---

## 3. Environment Requirements

- **Python**: recommended 3.9+ (no strict patch requirement)
- **Core dependencies**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `kagglehub`
  - `joblib`
  - `shap` (for interpretability)
- **Hardware**
  - CPU is sufficient (traditional ML models)

---

## 4. Quick Start

### 4.1 Install Dependencies (conda example)
```bash
conda create -n loan-approval python=3.10 -y
conda activate loan-approval

pip install pandas numpy matplotlib seaborn scikit-learn joblib shap kagglehub

# if needed
pip install notebook
````

### 4.2 Kaggle Access (if authentication is required)

`kagglehub` may require valid Kaggle credentials. A common setup:

* Create a Kaggle API token to get `kaggle.json`
* Place it at `~/.kaggle/kaggle.json` (and ensure permissions are correct)

### 4.3 Run the Notebook

```bash
jupyter notebook
```

Open and run: `Loan Approval Prediction.ipynb` (execute cells in order)

---

## 5. Example Outputs

The notebook will produce (exact numbers depend on random seeds and environment):

* Test-set metrics for each model (Accuracy / ROC-AUC / PR-AUC)
* Confusion matrices and classification reports
* PR curve plots
* Threshold scan curves and statistics at selected operating points
* Saved model and metrics files under `artifacts/`

---

## 6. FAQ

### Q1: Kaggle download fails / 403 / unauthorized

This usually means Kaggle credentials are missing or invalid. Follow the steps in **4.2 Kaggle Access** and retry.

### Q2: SHAP is very slow

`KernelExplainer` can be computationally expensive. Suggestions:

* Reduce the number of background samples and explained samples
* Skip SHAP cells first, and run them later once the core pipeline is verified

---

## 7. Project Structure

```
.
└── Loan Approval Prediction.ipynb
    └── (generated after running) artifacts/
        ├── rf_best_*.joblib
        ├── lr_best_*.joblib
        ├── svm_best_*.joblib
        ├── thresholds_*.json
        └── metrics_*.json
```
