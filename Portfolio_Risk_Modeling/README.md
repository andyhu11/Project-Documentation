# Portfolio Risk Modeling

> **A quantitative finance framework for Mean-Variance portfolio optimization and dynamic volatility forecasting.**

## 📖 Overview

**Portfolio Risk Modeling** is a computational finance toolkit built with **R**. It bridges the gap between Classical Portfolio Theory (Markowitz) and modern Time-Series Econometrics.

This project processes daily stock market data to construct optimal portfolios—specifically the **Global Minimum Variance (GMV)** and **Tangency** portfolios. Beyond static optimization, it employs **GARCH(1,1)** models to analyze the time-varying conditional volatility of these portfolios, providing a more realistic assessment of risk than standard deviation alone.

---

## ✨ Key Features

### 🛠 Data Processing & Statistics

* **Automated Cleaning:** Ingests raw OHLCV (Open-High-Low-Close-Volume) data, handling missing values and trading halts.
* **Log-Return Calculation:** Transforms price series into log-returns for statistical stationarity.
* **Statistical Testing:** Performs **Jarque-Bera tests** to check for normality and analyzes higher moments (Skewness and Kurtosis) to detect fat tails.

### 🧠 Optimization Engine (MPT)

* **Efficient Frontier Construction:** Simulates thousands of portfolio combinations to map the risk-return spectrum.
* **Quadratic Programming:** Uses `quadprog` to mathematically solve for exact weights of:
* **GMV Portfolio:** The theoretical portfolio with the lowest possible risk.
* **Tangency Portfolio:** The portfolio maximizing the Sharpe Ratio.



### 📉 Risk Modeling (Econometrics)

* **GARCH(1,1) Implementation:** Fits Generalized Autoregressive Conditional Heteroskedasticity models to portfolio returns.
* **Dynamic Volatility:** Extracts and visualizes conditional volatility () over time, capturing volatility clustering.

---

## 📂 Project Structure

```text
Portfolio_Risk_Modeling/
├── 📂 data/                                  # Raw and processed datasets
│   ├── daily_price_volume.csv                # Source: Raw OHLCV data
│   └── daily_price_volume_returns.csv        # Generated: Cleaned log-returns
│
├── 📂 output/                                # Generated visualizations & logs
│   ├── ConditionalVolatility_GARCH11.jpg     # Volatility time-series plots
│   ├── EfficientFrontier_Simulated.jpg       # Risk-Return frontier plots
│   ├── ReturnDistributions_Density.jpg       # Density comparison plots
│   └── DailyLogReturns_TimeSeries.jpg        # Return fluctuation plots
│
├── Portfolio_Risk_Modeling.R                 # Main analytical script
└── README.md                                 # Project documentation

```

---

## 🚀 Getting Started

### Prerequisites

* **R** (4.0 or newer recommended).
* **RStudio** (recommended for interactive visualization).
* **Required Packages:**
```r
install.packages(c("tidyverse", "PerformanceAnalytics", "moments", "quadprog", "rugarch", "gridExtra"))

```



### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Portfolio_Risk_Modeling.git

```


2. **Navigate to the project root:**
```bash
cd Portfolio_Risk_Modeling

```



### Usage Guide

1. **Prepare Data:**
Ensure `daily_price_volume.csv` is placed inside the `data/` directory.
2. **Run the Analysis:**
Open R or RStudio. Set your working directory to the project root and source the script directly:
```r
# Set working directory to project root
setwd("/path/to/Portfolio_Risk_Modeling")

# Run the main script
source("Portfolio_Risk_Modeling.R")

```


3. **View Outputs:**
* **Console:** Optimization weights (GMV & Tangency) will be printed directly.
* **Files:** Check the `output/` folder for generated high-resolution `.jpg` plots.



> **Note:** The script uses relative paths to load data (`./data/`) and save plots (`./output/`). Please ensure these folders exist before running.

---

## 📊 Visual Analytics

The project generates high-fidelity plots to aid in investment decision-making:

* **Efficient Frontier:** Visualizes the trade-off between risk (Std Dev) and Return, highlighting the optimal GMV and Tangency points against simulated portfolios.
* **Conditional Volatility:** Tracks how risk changes over time using GARCH(1,1), distinguishing between the stable GMV portfolio and the more volatile Tangency portfolio.
* **Return Density:** Compares the distribution of asset returns against a normal distribution to visualize "fat tails" and skewness.

---

## 🚧 Roadmap

The following modules are planned to extend the sophistication of the risk engine:

* **Backtesting Framework:** Implement a rolling-window backtest to validate the performance of the GMV and Tangency portfolios against a benchmark.
* **Alternative GARCH Models:** Incorporate **eGARCH** or **GJR-GARCH** to model the "leverage effect" (where negative returns increase volatility more than positive returns).
* **Value at Risk (VaR):** Automate the calculation of VaR and Expected Shortfall (ES) at 95% and 99% confidence intervals based on the GARCH forecasts.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewModel`).
3. Commit your Changes (`git commit -m 'Add some NewFeature'`).
4. Push to the Branch (`git push origin feature/NewModel`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
