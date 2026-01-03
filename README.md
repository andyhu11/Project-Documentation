# Project-Documentation

This repository is a collection of multiple independent projects (database system / machine learning / deep learning / financial risk modeling / time series forecasting / web scraping & data analysis). Each project lives in its own folder—open the folder to see its README, code, and reports.

---

## 📌 Projects at a Glance

| Project | Tech Stack | One-line Summary | Links |
|---|---|---|---|
| Books Management System | Microsoft Access (.accdb) | An Access-based book management system for purchasing, inventory, sales logging, and reporting (with a user guide). | [Folder](./Books_Management_System) · [README](./Books_Management_System/README.md) |
| Image Classification (CNN) | PyTorch / torchvision | CIFAR-10 image classification: augmentation, CNN training, best-model saving, evaluation, and visualization. | [Folder](./Image_Classification_CNN) · [README](./Image_Classification_CNN/README.md) |
| Loan Approval Prediction | scikit-learn / SHAP | Loan approval binary classification: EDA, leakage-free preprocessing, feature engineering, tuning, threshold analysis, and model artifacts. | [Folder](./Loan_Approval_Prediction) · [README](./Loan_Approval_Prediction/README.md) |
| Portfolio Risk Modeling | R / copula / rugarch | Markowitz portfolio optimization + Copula-based VaR + GARCH volatility & VaR (with plots and outputs). | [Folder](./Portfolio_Risk_Modeling) · [README](./Portfolio_Risk_Modeling/README.md) |
| Time Series Forecasting (Solar Prosumers) | XGBoost / LSTM / Optuna / TensorFlow | Forecasting prosumers’ electricity consumption and solar generation using XGBoost and LSTM (optional Attention), with a full report. | [Folder](./Time_Series_Forecasting) · [README](./Time_Series_Forecasting/README.md) |
| Web Scraping & Data Analysis (TVmaze) | requests / BeautifulSoup / statsmodels | Scrape metadata for 200 TV shows from TVmaze to CSV, then run statistical tests and robust regression analyses. | [Folder](./Web_Scraping_Data_Analysis) · [README](./Web_Scraping_Data_Analysis/README.md) |

---

## 1) Books Management System (Microsoft Access)

- **What it is**: A Microsoft Access `.accdb` project with tables/queries/forms/reports for common workflows (purchase, inventory, sales, and analytics).
- **Highlights**:
  - Default login: `admin / admin`
  - Recommended: add the folder to Access **Trusted Locations** to avoid blocked macros/content
- **Entry**:
  - [Project folder](./Books_Management_System)
  - [Project README](./Books_Management_System/README.md)

---

## 2) Image Classification with CNN (CIFAR-10 / PyTorch)

- **What it is**: A full notebook pipeline: environment checks → data loading/preprocessing → CNN modeling → train/val → save best model → test evaluation → inference visualization.
- **Highlights**:
  - Dataset: CIFAR-10 (10 classes, 32×32 RGB images)
  - Best checkpoint: `best_cifar10_cnn.pt`
- **Entry**:
  - [Project folder](./Image_Classification_CNN)
  - [Project README](./Image_Classification_CNN/README.md)

---

## 3) Loan Approval Prediction (Binary Classification / sklearn)

- **What it is**: End-to-end ML project: EDA, visualization, leakage-free preprocessing, feature engineering, multi-model tuning, SHAP explainability, threshold strategy analysis, and exporting artifacts.
- **Highlights**:
  - Models compared: Logistic Regression / Random Forest / SVM (ROC-AUC as the main tuning metric)
  - Outputs saved under `./artifacts/` (joblib model, threshold json, metrics json)
- **Entry**:
  - [Project folder](./Loan_Approval_Prediction)
  - [Project README](./Loan_Approval_Prediction/README.md)

---

## 4) Portfolio Risk Modeling (Markowitz + Copula VaR + GARCH / R)

- **What it is**: Uses daily stock returns to perform mean-variance portfolio analysis (GMV/tangency/constrained portfolios), Copula VaR via Monte Carlo + rolling backtests, and GARCH(1,1) conditional volatility & VaR.
- **Highlights**:
  - Multiple key plots: return series, distributions, efficient frontier, conditional volatility, etc.
  - Clear structure for data and outputs (`data/`, `output_picture/`)
- **Entry**:
  - [Project folder](./Portfolio_Risk_Modeling)
  - [Project README](./Portfolio_Risk_Modeling/README.md)

---

## 5) Time Series Forecasting (Solar Panel Prosumers)

- **What it is**: Forecasts both consumption and generation for solar prosumers using two model families: XGBoost and LSTM (optional Attention). Includes a PDF report summarizing results.
- **Highlights**:
  - Unified metrics: MSE / RMSE / MAE / R² / MAPE
  - XGBoost tuning with Optuna + TimeSeriesSplit; LSTM uses sliding windows and training strategies (e.g., early stopping)
- **Entry**:
  - [Project folder](./Time_Series_Forecasting)
  - [Project README](./Time_Series_Forecasting/README.md)

---

## · Web Scraping & Data Analysis (TVmaze)

- **What it is**: Notebook in two parts:
  1) Scrape 200 TV shows from TVmaze (collect links → crawl metadata → export fixed-schema CSV)
  2) Read CSV for feature engineering + statistical analysis (e.g., Kruskal–Wallis, Dunn post-hoc, Mann–Whitney U, HC3 robust regression)
- **Highlights**:
  - Fixed output schema:
    `Title, First air date, End date, Rating, Genres, Status, Network, Summary`
  - Polite scraping practices (delay/retry strategies)
- **Entry**:
  - [Project folder](./Web_Scraping_Data_Analysis)
  - [Project README](./Web_Scraping_Data_Analysis/README.md)
