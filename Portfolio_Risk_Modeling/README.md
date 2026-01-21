# Portfolio Risk Modeling

![R](https://img.shields.io/badge/Language-R_4.0%2B-blue)
![Library](https://img.shields.io/badge/Library-rugarch%20%7C%20quadprog-orange)
![Methodology](https://img.shields.io/badge/Methodology-MPT%20%7C%20GARCH-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> **A quantitative finance framework for Mean-Variance portfolio optimization and dynamic volatility forecasting.**

## 📖 Overview

**Portfolio Risk Modeling** is a computational finance toolkit built with **R**. It bridges the gap between Classical Portfolio Theory (Markowitz) and modern Time-Series Econometrics.

This project processes daily stock market data to construct optimal portfolios—specifically the **Global Minimum Variance (GMV)** and **Tangency** portfolios. Beyond static optimization, it employs **GARCH(1,1)** models to analyze the time-varying conditional volatility of these portfolios, providing a more realistic assessment of risk than standard deviation alone.

---

## ✨ Key Features

### 🛠 Data Engineering & Statistics
* **Log-Return Transformation:** Automatically transforms raw OHLCV price series into log-returns to ensure statistical stationarity.
* **Statistical Diagnostics:**
    * Integrates the **Jarque-Bera Test** for normality checks.
    * Computes higher moments (**Skewness** and **Kurtosis**) to detect "fat tail" characteristics in financial data.
* **Automated Cleaning:** Intelligently handles missing values and data alignment issues caused by trading suspensions.

### 🧠 Convex Optimization
* **Quadratic Programming:** Utilizes the `solve.QP` solver from the `quadprog` library to calculate exact weights under constraints.
    * **GMV Portfolio:** The theoretical portfolio with the lowest possible risk.
    * **Tangency Portfolio:** The portfolio maximizing the Sharpe Ratio (Risk-Adjusted Return) under the risk-free rate constraint.
* **Efficient Frontier:** Generates the risk-return efficient frontier through Monte Carlo Simulation, mapping thousands of potential portfolio combinations.

### 📉 Econometric Modeling
* **GARCH(1,1) Integration:** Models the Generalized Autoregressive Conditional Heteroskedasticity of portfolio returns using the `rugarch` library.
* **Volatility Clustering:** Captures volatility clustering in financial time series, extracting dynamic Sigma ($\sigma_t$) rather than relying on static historical volatility.

---

## 📂 Project Structure

```text
Portfolio_Risk_Modeling/
├── 📂 data/                          # Raw and processed datasets
│   ├── daily_price_volume.csv        # Source: Raw OHLCV data
│   └── daily_price_volume_returns.csv# Generated: Cleaned log-returns
│
├── 📂 output/                        # Generated visualizations & logs
│   ├── ConditionalVolatility_GARCH11.jpg # Volatility time-series plots
│   ├── EfficientFrontier_Simulated.jpg   # Risk-Return frontier plots
│   ├── ReturnDistributions_Density.jpg   # Density comparison plots
│   └── DailyLogReturns_TimeSeries.jpg    # Return fluctuation plots
│
├── Portfolio_Risk_Modeling.R         # Main analytical script
└── README.md                         # Project documentation

```

---

## 📊 Visual Analytics

The project generates high-fidelity plots to aid in quantitative decision-making:

### 1. Efficient Frontier

Visualizes the trade-off between risk (Std Dev) and Return, highlighting the optimal GMV and Tangency points against simulated portfolios.

### 2. Conditional Volatility

Tracks how risk changes over time using GARCH(1,1), distinguishing between the stable GMV portfolio and the more volatile Tangency portfolio, identifying periods of market turmoil.

### 3. Return Density

Compares the distribution of asset returns against a standard normal distribution to visualize "fat tails" (Leptokurtosis) and skewness.

---

## 🚀 Getting Started

### Prerequisites

* **R** (version 4.0 or newer)
* **RStudio** (recommended)
* **Required Libraries:**
Ensure all core libraries, including `rugarch`, are installed:

```r
install.packages(c("tidyverse", "PerformanceAnalytics", "moments", "quadprog", "rugarch", "gridExtra"))

```

### Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/your-username/Portfolio_Risk_Modeling.git](https://github.com/your-username/Portfolio_Risk_Modeling.git)

```

2. **Navigate to the project root:**

```bash
cd Portfolio_Risk_Modeling

```

### Usage Guide

1. **Prepare Data:**
Ensure your raw data `daily_price_volume.csv` is placed inside the `data/` directory.
2. **Run Analysis:**
Open RStudio, set the working directory to the project root, and source the script:

```r
# Set working directory
setwd("/path/to/Portfolio_Risk_Modeling")

# Execute the pipeline
source("Portfolio_Risk_Modeling.R")

```

3. **Interpret Outputs:**
* **Console Output:** The script will print the optimal weight vectors (, ) and GARCH model summaries.
* **Visualizations:** High-resolution plots will be saved to the `output/` folder.



---

## 🚧 Roadmap

* **Backtesting Engine:** Implement rolling-window backtesting to evaluate out-of-sample performance.
* **Risk Metrics:** Automate calculation of **Value at Risk (VaR)** and **Expected Shortfall (ES)** based on GARCH forecasts.
* **Advanced GARCH:** Incorporate **eGARCH** or **GJR-GARCH** to model the leverage effect (asymmetric volatility response).

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewAlgorithm`).
3. Commit your Changes (`git commit -m 'Add eGARCH support'`).
4. Push to the Branch (`git push origin feature/NewAlgorithm`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

```

```
