# Solar Prosumer Energy Forecasting

> **A machine learning pipeline utilizing LSTM and XGBoost to predict electricity consumption and production behaviors of solar panel prosumers.**

## 📖 Overview

**Solar Prosumer Energy Forecasting** is a predictive analytics project designed to address the energy imbalance issues inherent in modern smart grids. Leveraging the **Enefit** dataset, this project aims to minimize the "energy imbalance cost" by accurately forecasting how prosumers (consumers who also produce energy) behave.

The solution moves beyond traditional statistical baselines (ARIMA/SARIMA), implementing advanced Deep Learning (**LSTM**) and Gradient Boosting (**XGBoost**) techniques. Following the **CRISP-DM** methodology, the project covers the full data science lifecycle—from rigorous feature engineering and scaling to hyperparameter optimization via Optuna.

---

## ✨ Key Features

### 🧠 Advanced Modeling Architectures

* **Long Short-Term Memory (LSTM):**
* Custom-built Recurrent Neural Networks (RNN) designed to capture long-term temporal dependencies in energy usage.
* Implemented using **TensorFlow/Keras** with callbacks for Early Stopping and Learning Rate Reduction.


* **eXtreme Gradient Boosting (XGBoost):**
* High-performance ensemble models optimized for tabular time-series data.
* Features automated hyperparameter tuning to maximize  scores.



### 🛠 Data Engineering Pipeline

* **Robust Preprocessing:**
* Data cleaning and imputation strategies for handling missing weather and meter data.
* **Feature Scaling:** Standardization of input variables (seen in `final_data_for_consumption_scaled.csv`) to ensure model stability.


* **Temporal Feature Extraction:**
* Cyclical encoding of time features (e.g., `sin_hour`, `cos_doy`) to preserve the periodic nature of daily and seasonal energy cycles.



### 📊 Model Optimization & Evaluation

* **Hyperparameter Tuning:** Integration with **Optuna** for automated search of optimal model parameters (learning rates, estimators, depth).
* **Comprehensive Metrics:** Evaluation based on industry-standard metrics: **RMSE**, **MAE**, ****, and **MAPE**.

---

## 📈 Performance

Based on the empirical analysis of the **Enefit** dataset, the tuned XGBoost model demonstrated superior predictive capabilities compared to both deep learning and statistical baselines.

### 1. Evaluation Results (Mean Metrics)

| Model Category | Task |  Score | RMSE | MAE |
| --- | --- | --- | --- | --- |
| **SARIMAX (Baseline)** | Consumption | 0.7994 | 0.5302 | 0.4846 |
|  | Production | 0.8358 | 0.2815 | 0.1541 |
| **XGBoost (Tuned)** | Consumption | **0.9567** | **0.0534** | **0.0346** |
|  | Production | **0.9686** | **0.0422** | **0.0137** |
| **LSTM (Tuned)** | Consumption | 0.8938 | 0.0654 | 0.0437 |
|  | Production | 0.9552 | 0.0559 | 0.0276 |

### 2. Key Technical Insights

* **Optimization Gain:** Hyperparameter tuning via Optuna improved the XGBoost consumption  from 0.9457 to **0.9567**.
* **Driver Analysis:** * **Production:** Highly sensitive to surface solar radiation, with cloud cover showing a secondary effect.
* **Consumption:** Heavily influenced by historical lag features (1-hour prior usage), whereas real-time electricity prices had a lower immediate correlation.


* **Architecture Comparison:** While LSTM successfully captured long-term dependencies, XGBoost proved more robust for the structured, high-dimensional tabular time-series features in this specific dataset.

---

## 📂 Project Structure

```text
Time_Series_Forecasting/
├── Data/
│   ├── final_data_for_consumption_scaled.csv   # Pre-processed/Scaled consumption data
│   └── final_data_for_production_scaled.csv    # Pre-processed/Scaled production data
├── Models/
│   ├── LSTM_consumption_model.py               # Deep Learning training script (Consumption)
│   ├── LSTM_production_model.py                # Deep Learning training script (Production)
│   ├── xgboost_consumption_model.py            # Gradient Boosting script (Consumption)
│   └── xgboost_production_model.py             # Gradient Boosting script (Production)
├── Time Series Forecasting of Energy Behavior in Solar Panel Prosumers.pdf # Full Project Report
└── README.md                                   # Project Documentation

```

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.8+**
* **TensorFlow** (2.x)
* **XGBoost**
* **Optuna** (for optimization)
* **Pandas / NumPy / Scikit-Learn**

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Project-Documentation.git

```

2. Navigate to the project directory:

```bash
cd Project-Documentation/Time_Series_Forecasting

```

3. Install dependencies:

```bash
pip install pandas numpy tensorflow xgboost scikit-learn optuna

```

### Usage Guide

1. **Data Preparation:** Ensure dataset files are in `Data/`.
2. **Training Models:**

```bash
cd Models
# Run LSTM Training
python LSTM_consumption_model.py
# Run XGBoost Optimization
python xgboost_production_model.py

```

> **Note:** For a deep dive into the mathematical theory, feature importance analysis, and result interpretation, please refer to the **[Project Report PDF](https://github.com/andyhu11/Project-Documentation/blob/main/Time_Series_Forecasting/Time%20Series%20Forecasting%20of%20Energy%20Behavior%20in%20Solar%20Panel%20Prosumers.pdf)**.

---

## 🚧 Roadmap & Future Enhancements

* **Model Fusion Strategy:** Implement a weighted ensemble of LSTM and XGBoost to further reduce variance.
* **Real-Time Inference API:** Wrap models in **FastAPI** for real-time grid operator support.
* **Transformer Architecture:** Experiment with **Temporal Fusion Transformers (TFT)** for better handling of static metadata.

---

## 🤝 Contributing

Contributions are welcome:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewArchitecture`).
3. Commit Changes (`git commit -m 'Add Transformer Model'`).
4. Push to the Branch.
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
