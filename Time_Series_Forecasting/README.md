# Time Series Forecasting of Energy Behavior in Solar Panel Prosumers

> This project focuses on solar prosumers (households that both **consume** and **produce** electricity).  
> It provides time-series forecasting pipelines for **consumption** and **production**, using two model families: **XGBoost** and **LSTM (with optional Attention)**.  
> The repository includes two CSV datasets, four model scripts, and a full explanatory PDF report (which concludes XGBoost performs best overall on both tasks).

---

## 1. Project Overview

This repository implements an end-to-end workflow from **data → modeling → tuning → evaluation → exporting results**.

- **Targets**
  - Consumption forecasting: `target_cons`
  - Production forecasting: `target_prod`
- **Model approaches**
  - **XGBoost**: strong at learning non-linear relationships and feature interactions
  - **LSTM**: effective at capturing temporal dependencies in time-series data
- **Metrics**: MSE / RMSE / MAE / R² / MAPE (reported and saved by the scripts)
- **Key findings (from the report)**
  - Production is mainly driven by solar irradiance; consumption depends more on past usage patterns
  - Overall, XGBoost achieves the best performance across both consumption and production tasks

---

## 2. Repository Structure

```text
.
├── README.md
├── data/
│   ├── final_data_for_consumption_scaled.csv
│   └── final_data_for_production_scaled.csv
├── Model/
│   ├── xgboost_consumption_model.py
│   ├── xgboost_production_model.py
│   ├── LSTM_consumption_model.py
│   └── LSTM_production_model.py
└── Time Series Forecasting of Energy Behavior in Solar Panel Prosumers.pdf
````

---

## 3. Features

### 3.1 LSTM (Consumption / Production)

* **Sliding-window sequence generation**: `time_steps = 24` (use the past 24 hours to predict the next hour)
* **Chronological split**: 70% / 15% / 15% for train / validation / test
* **Model architecture**

  * Multi-layer LSTM + Dropout (configurable layers, hidden size, dropout)
  * Optional Self-Attention + GlobalAveragePooling1D
* **Training strategy**

  * 5-fold `TimeSeriesSplit` cross-validation to select the best configuration
  * EarlyStopping + ReduceLROnPlateau + ModelCheckpoint (saves the best model)
* **Outputs**

  * Cross-validation results: `cv_results_*_lstm.csv`
  * Best checkpoint: `best_lstm_model.keras`
  * Test metrics: `best_test_metrics_*_lstm.csv`
* **Result directories**

  * Consumption: `./results/consumption_baseline`
  * Production: `./results/production_baseline`

---

### 3.2 XGBoost (Consumption / Production)

* **Chronological split**: 70% / 15% / 15% (train / val / test)
* **Baseline model**: `XGBRegressor(n_estimators=100, random_state=42)`
* **Unified evaluation function**: reports and returns MSE / RMSE / MAE / R² / MAPE
* **Optuna hyperparameter tuning + 5-fold time-series CV**

  * Uses mean CV R² as the optimization objective for both tasks
* **Exports**

  * Timestamped results folder: `results_{timestamp}`
  * Best model saved as JSON: `best_model_*_90plus_{timestamp}.json`
  * Performance report exported to Excel: `model_performance_*_90plus_{timestamp}.xlsx`

---

### 3.3 PDF Report

The PDF provides the complete methodology and results, covering data preparation, feature design, model comparisons, and conclusions. It also explains why XGBoost and LSTM were introduced to handle non-linearity and temporal dependencies beyond classical ARIMA/SARIMA-style approaches.

---

## 4. Environment Requirements

* **Python**: 3.9+ (recommended 3.10)
* **Core dependencies**

  * `pandas`, `numpy`
  * `scikit-learn`
  * `xgboost`
  * `optuna`
  * `tensorflow` (Keras)
  * `openpyxl` (for exporting `.xlsx`)

---

## 5. Quick Start

### 5.1 Install dependencies (pip example)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install numpy pandas scikit-learn xgboost optuna tensorflow openpyxl
```

### 5.2 Prepare data

Place the datasets under `data/`:

* `data/final_data_for_consumption_scaled.csv`
* `data/final_data_for_production_scaled.csv`

---

## 6. Outputs

### LSTM

* `results/consumption_baseline/`

  * `cv_results_consumption_lstm.csv`
  * `best_lstm_model.keras`
  * `best_test_metrics_consumption_lstm.csv`
* `results/production_baseline/` (same structure)

### XGBoost

* `results_YYYYMMDD_HHMMSS/`

  * `best_model_cons_90plus_*.json` / `best_model_prod_90plus_*.json`
  * `model_performance_*_90plus_*.xlsx`

---

## 7. FAQ

### Q1: TensorFlow / CUDA issues

* The code runs on CPU as well (just slower).
* If you use GPU, ensure your TensorFlow build is compatible with your CUDA/cuDNN setup (or install an official GPU-supported TensorFlow package for your platform).

---

## 8. References

* Full methodology and conclusions are documented in:
  `Time Series Forecasting of Energy Behavior in Solar Panel Prosumers.pdf`
