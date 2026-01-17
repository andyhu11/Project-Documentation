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
* **Comprehensive Metrics:** Evaluation based on industry-standard metrics:
* **RMSE** (Root Mean Square Error)
* **MAE** (Mean Absolute Error)
* **$R^2$** (Coefficient of Determination)
* **MAPE** (Mean Absolute Percentage Error)



---

## 📂 Project Structure

```text
Time_Series_Forecasting/
├── Data/
│   ├── final_data_for_consumption_scaled.csv   # Pre-processed/Scaled consumption data
│   └── final_data_for_production_scaled.csv    # Pre-processed/Scaled production data
├── Models/
│   ├── LSTM_consumption_model.py               # Deep Learning training script (Consumption)
│   ├── LSTM_production_model.py                # Deep Learning training script (Production)
│   ├── xgboost_consumption_model.py            # Gradient Boosting script (Consumption)
│   └── xgboost_production_model.py             # Gradient Boosting script (Production)
├── Time Series Forecasting of Energy Behavior.pdf # Full Project Report & Methodology
└── README.md                                   # Project Documentation

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

1. **Data Preparation:**
Ensure your dataset files are located in the `Data/` directory as shown in the project structure. The scripts are configured to read from this relative path.
2. **Training Models:**
Navigate to the `Models/` directory to run the training scripts:
```bash
cd Models

```


3. **Run LSTM Training:**
```bash
python LSTM_consumption_model.py

```


*This will initialize the neural network, train over 50 epochs (default), and save the best model artifact.*
4. **Run XGBoost Optimization:**
```bash
python xgboost_production_model.py

```


*This script initiates the Optuna study to find the best hyperparameters before training the final booster.*

> **Note:** For a deep dive into the mathematical theory, feature importance analysis, and result interpretation, please refer to the **[Project Report](https://www.google.com/search?q=./Time%2520Series%2520Forecasting%2520of%2520Energy%2520Behavior.pdf)** located in the root of this folder.

---

## 🚧 Roadmap & Future Enhancements

The current implementation focuses on offline batch forecasting. Future iterations aim to productionize the workflow:

* **Model Fusion Strategy:**
* Implement a weighted ensemble of the LSTM and XGBoost outputs to reduce variance and improve generalization on unseen data.


* **Real-Time Inference API:**
* Wrap the trained models in a **FastAPI** or **Flask** container to serve predictions to grid operators in real-time.


* **Transformer Architecture:**
* Experiment with **Temporal Fusion Transformers (TFT)** to better handle static covariates (metadata) alongside time-varying inputs.



---

## 🤝 Contributing

Contributions to improve model accuracy or efficiency are welcome:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewArchitecture`).
3. Commit your Changes (`git commit -m 'Add Transformer Model'`).
4. Push to the Branch (`git push origin feature/NewArchitecture`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
