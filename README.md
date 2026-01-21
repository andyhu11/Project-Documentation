# 🚀 Engineering & Data Science Portfolio

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/Language-R-276DC3?logo=r&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![SQL](https://img.shields.io/badge/Database-SQL-4479A1?logo=postgresql&logoColor=white)

> **A comprehensive collection of technical projects demonstrating end-to-end expertise in Machine Learning, Quantitative Finance, and Full-Stack Data Engineering.**

## 📂 Repository Overview

This repository serves as a centralized portfolio containing six production-grade projects. Each directory represents a standalone application or research pipeline, complete with source code, documentation, and rigorous performance analysis.

### 🧠 Deep Learning & Time Series
| Project | Domain | Tech Stack | Key Impact |
| :--- | :--- | :--- | :--- |
| **[Solar Energy Forecasting](./Time_Series_Forecasting)** | Smart Grid / Energy | `XGBoost` `LSTM` `Optuna` | Reduced MAPE error significantly vs. ARIMA baselines; engineered hybrid forecasting models for prosumer consumption/production. |
| **[CNN Image Classification](./Image_Classification_CNN)** | Computer Vision | `PyTorch` `torchvision` | Achieved **84.6% Accuracy** on CIFAR-10 using a custom 3-layer CNN with adaptive pooling and augmentation pipelines. |

### 🤖 Classical Machine Learning
| Project | Domain | Tech Stack | Key Impact |
| :--- | :--- | :--- | :--- |
| **[Loan Approval Prediction](./Loan_Approval_Prediction)** | FinTech / Risk | `Scikit-Learn` `SHAP` `Random Forest` | Built an automated underwriting system with **99.5% Precision** and **100% Recall**; integrated SHAP for regulatory explainability. |
| **[TV Show Analytics](./Web_Scraping_Data_Analysis)** | Data Mining | `SciPy` `BeautifulSoup` `Statsmodels` | End-to-end scraper for 200+ shows; applied Kruskal-Wallis & Robust Regression to debunk "Golden Age" TV myths. |

### 📉 Quantitative Finance & Systems
| Project | Domain | Tech Stack | Key Impact |
| :--- | :--- | :--- | :--- |
| **[Portfolio Risk Modeling](./Portfolio_Risk_Modeling)** | Quant Finance | `R` `GARCH` `Copula` | Implemented Mean-Variance optimization (Markowitz) and Dynamic Volatility forecasting using GARCH(1,1). |
| **[UniBooks System](./Books_Management_System)** | DBMS | `MS Access` `VBA` `SQL` | Designed a normalized relational database with RBAC security and automated inventory tracking triggers. |

---

## 🛠 Technical Deep Dives

### 1. Solar Prosumer Energy Forecasting
* **Challenge:** Mitigate energy imbalance costs in smart grids by predicting erratic prosumer behavior.
* **Solution:** Developed a comparative pipeline using **Gradient Boosting (XGBoost)** and **Recurrent Neural Networks (LSTM)**.
* **Highlights:**
    * Automated hyperparameter tuning via **Optuna** (Bayesian Optimization).
    * Implemented 5-fold `TimeSeriesSplit` cross-validation to prevent look-ahead bias.
    * **Artifacts:** Full technical report (`.pdf`) and production-ready Python scripts.
* 👉 **[View Project](./Time_Series_Forecasting)**

### 2. Loan Approval AI & Fairness
* **Challenge:** Automate loan eligibility while minimizing financial risk and maintaining interpretability.
* **Solution:** A **Random Forest** classifier tuned for high precision in the "Safe-to-Approve" band.
* **Highlights:**
    * **Feature Engineering:** Created high-impact ratios (e.g., Debt-to-Income, Asset Liquidity).
    * **Governance:** Utilized **SHAP** (SHapley Additive exPlanations) to audit model decisions for bias.
    * **Performance:** Achieved ROC-AUC of **0.999** on the test set.
* 👉 **[View Project](./Loan_Approval_Prediction)**

### 3. Quantitative Risk Engine (R)
* **Challenge:** Model portfolio risk beyond simple standard deviation in volatile markets.
* **Solution:** A statistical framework combining **Modern Portfolio Theory (MPT)** with time-series econometrics.
* **Highlights:**
    * **Convex Optimization:** Calculated Global Minimum Variance (GMV) and Tangency portfolios using quadratic programming.
    * **Volatility Modeling:** Integrated **GARCH(1,1)** to capture volatility clustering and "fat tails" in asset returns.
    * **Backtesting:** Rolling-window analysis to validate Value-at-Risk (VaR) estimations.
* 👉 **[View Project](./Portfolio_Risk_Modeling)**

### 4. CNN Image Classification (Computer Vision)
* **Challenge:** Implement a robust vision pipeline from scratch without relying on pre-trained models.
* **Solution:** Designed a custom **3-layer Convolutional Neural Network (CNN)** for the CIFAR-10 dataset.
* **Highlights:**
    * **Architecture:** Utilized `Conv2d` blocks with Batch Normalization and Max Pooling; integrated **Dropout** to prevent overfitting.
    * **Augmentation:** Applied random rotations and horizontal flips to improve generalization.
    * **Result:** Achieved **84.6% Accuracy**, with strong performance on mechanical classes (Cars/Trucks).
* 👉 **[View Project](./Image_Classification_CNN)**
---

## ⚡ Getting Started

Each project is self-contained. To run a specific project:

1.  **Navigate** to the project folder.
2.  **Read** the local `README.md` for specific dependency installation (e.g., `pip install -r requirements.txt` or R library installation).
3.  **Launch** the corresponding Jupyter Notebook (`.ipynb`) or R Script (`.R`).

```bash
# Example: Cloning the repo
git clone [https://github.com/your-username/Project-Documentation.git](https://github.com/your-username/Project-Documentation.git)
cd Project-Documentation

```

---

## 📄 License

This repository is licensed under the **MIT License**. See individual project folders for specific third-party attributions.
