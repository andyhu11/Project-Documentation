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

## Books Management System (Microsoft Access)

- **What it is**: A Microsoft Access `.accdb` project with tables/queries/forms/reports for common workflows (purchase, inventory, sales, and analytics).
- **Highlights**:
  - End-to-end workflow: purchasing → inventory updates → sales logging → summary reports
  - Real-time inventory updates to track stock levels and changes promptly
- **Entry**:
  - [Project folder](./Books_Management_System)
  - [Project README](./Books_Management_System/README.md)

---

## Image Classification with CNN (CIFAR-10 / PyTorch)

- **What it is**: A full notebook pipeline: environment checks → data loading/preprocessing → CNN modeling → train/val → save best model → test evaluation → inference visualization.
- **Highlights**:
  - Training curves and evaluation metrics are tracked to compare model behavior across epochs
  - Includes qualitative visualization (sample predictions) to quickly sanity-check model performance
- **Entry**:
  - [Project folder](./Image_Classification_CNN)
  - [Project README](./Image_Classification_CNN/README.md)

---

## Loan Approval Prediction (Binary Classification / sklearn)

- **What it is**: End-to-end ML project: EDA, visualization, leakage-free preprocessing, feature engineering, multi-model tuning, SHAP explainability, threshold strategy analysis, and exporting artifacts.
- **Highlights**:
  - Models compared: Logistic Regression / Random Forest / SVM (ROC-AUC as the main tuning metric)
  - Threshold policy analysis: scans decision thresholds and reports operating points (e.g., cautious vs. balanced) with precision/recall/F1 trade-offs
- **Entry**:
  - [Project folder](./Loan_Approval_Prediction)
  - [Project README](./Loan_Approval_Prediction/README.md)

---

## Portfolio Risk Modeling (Markowitz + Copula VaR + GARCH / R)

- **What it is**: Uses daily stock returns to perform mean-variance portfolio analysis (GMV/tangency/constrained portfolios), Copula VaR via Monte Carlo + rolling backtests, and GARCH(1,1) conditional volatility & VaR.
- **Highlights**:
  - Copula-based VaR via Monte Carlo, including rolling-window backtesting with breach counting and a binomial test
  - Automatically engineers log returns from raw prices and exports a reusable returns dataset for downstream modeling
- **Entry**:
  - [Project folder](./Portfolio_Risk_Modeling)
  - [Project README](./Portfolio_Risk_Modeling/README.md)

---

## Time Series Forecasting (Solar Panel Prosumers)

- **What it is**: Forecasts both consumption and generation for solar prosumers using two model families: XGBoost and LSTM (optional Attention). Includes a PDF report summarizing results.
- **Highlights**:
  - XGBoost uses Optuna with 5-fold TimeSeriesSplit for hyperparameter search, and exports the best model as JSON plus a performance report to Excel (with timestamped result folders)
  - LSTM builds sequences with a 24-hour sliding window, supports optional Self-Attention, and trains with EarlyStopping / ReduceLROnPlateau / ModelCheckpoint alongside 5-fold TimeSeriesSplit cross-validation
- **Entry**:
  - [Project folder](./Time_Series_Forecasting)
  - [Project README](./Time_Series_Forecasting/README.md)

---

## Web Scraping & Data Analysis (TVmaze)

- **What it is**: Notebook in two parts:
  1) Scrape 200 TV shows from TVmaze (collect links → crawl metadata → export fixed-schema CSV)
  2) Read CSV for feature engineering + statistical analysis (e.g., Kruskal–Wallis, Dunn post-hoc, Mann–Whitney U, HC3 robust regression)
- **Highlights**:
  - Implements polite scraping practices (request throttling / retry logic) to reduce failure rates
  - Includes statistical testing + robust regression to validate relationships beyond simple correlations
- **Entry**:
  - [Project folder](./Web_Scraping_Data_Analysis)
  - [Project README](./Web_Scraping_Data_Analysis/README.md)
