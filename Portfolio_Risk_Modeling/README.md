# Portfolio Risk Modeling (Markowitz + Copula VaR + GARCH)

> A complete **R** project that covers: **return engineering → mean-variance portfolio optimization (GMV / Tangency) → Copula-based VaR → GARCH conditional volatility & VaR**.  
> Main script: `Portfolio_Risk_Modeling.R`

---

## 1. Project Overview

This project uses daily data for **five stocks** and implements the following workflow:

1) Compute **daily log returns** from price data and build a wide-format return matrix aligned by date.  
2) Produce **descriptive statistics**, and visualize return time series and distributions.  
3) Mean-variance (Markowitz) portfolio analysis:
   - Simulate an efficient frontier (unconstrained random portfolios)
   - Compute **GMV** and **Tangency** portfolios in closed form
   - Simulate **constrained portfolios** (no shorting, max weight cap)  
4) **Gaussian Copula** Monte Carlo simulation for portfolio returns, then compute VaR and run rolling VaR backtesting.  
5) Fit **GARCH(1,1)** models to GMV / Tangency / optimal portfolio returns, plot conditional volatilities, and compute 1-day 95% VaR using forecasted volatility.

---

## 2. Folder Structure

```text
.
├─ Portfolio_Risk_Modeling.R
├─ data/
│  ├─ daily_price_volume.csv
│  └─ daily_price_volume_returns.csv
└─ output_picture/
   ├─ DailyLogReturns_TimeSeries.jpg
   ├─ ReturnDistributions_DailyLogReturns_Density.jpg
   ├─ EfficientFrontier_Simulated_GMV_Tangency.jpg
   └─ ConditionalVolatility_GARCH11_GMV_Tangency.jpg
````

### 2.1 Data Files

* `daily_price_volume.csv`: raw daily dataset (the script reads this file and converts `Trddt` to `Date`).
* `daily_price_volume_returns.csv`: returns dataset exported by the script (includes `LogRet` by stock and date).

### 2.2 Output Figures

* `DailyLogReturns_TimeSeries.jpg`: faceted time series plot of daily log returns for the 5 assets.
* `ReturnDistributions_DailyLogReturns_Density.jpg`: histogram + kernel density of returns (faceted).
* `EfficientFrontier_Simulated_GMV_Tangency.jpg`: simulated efficient frontier scatter with GMV and Tangency labeled.
* `ConditionalVolatility_GARCH11_GMV_Tangency.jpg`: GARCH(1,1) conditional volatility series for GMV and Tangency portfolios.

---

## 3. Features

### 3.1 Return Construction & Descriptive Analysis (Question 1a)

* Selects 5 tickers: `678, 868, 600741, 600006, 600151`
* Computes daily log returns from adjusted close price `Adjprcnd`:

  * `LogRet = log(AdjClose) - log(lag(AdjClose))`
* Builds a wide return matrix `ret_matrix` (rows = dates, cols = assets)
* Computes summary statistics: mean, median, std, skewness, kurtosis, quantiles, etc.
* Visualizations:

  * Faceted return time series
  * Faceted return distribution (hist + density)

### 3.2 Mean-Variance Portfolios & Efficient Frontier (Question 1b)

* Simulates 10,000 unconstrained random portfolios and computes:

  * portfolio return / standard deviation / Sharpe ratio
* Closed-form solutions:

  * **GMV**: ( w \propto \Sigma^{-1}\mathbf{1} )
  * **Tangency**: ( w \propto \Sigma^{-1}(\mu-r_f) ) (the script uses `rf = 0`)
* Plots the simulated efficient frontier and highlights GMV/Tangency.

### 3.3 Constrained Portfolios (No shorting & weight cap)

* Constraints: `weight >= 0` and `weight <= 0.25`
* Randomly samples portfolios and filters feasible weights
* Selects:

  * constrained GMV (minimum volatility)
  * constrained Tangency (maximum Sharpe ratio)

### 3.4 Copula-based VaR (Question 2)

* Maps each asset return series to ([0,1]) using empirical ranks and fits a **Gaussian copula**
* Runs copula Monte Carlo simulation (e.g., `n_sim = 1e5`), maps simulated uniforms back to returns via empirical quantiles, then computes portfolio VaR
* Compares multiple VaR methods:

  * Copula VaR
  * Variance-Covariance (Normal approximation)
  * Historical VaR
* Rolling-window VaR (window = 1000, alpha = 0.10) + breach counting and binomial backtest

### 3.5 GARCH(1,1) Conditional Volatility & VaR (Question 3)

* Fits `rugarch` sGARCH(1,1) with Normal innovations for GMV and Tangency portfolio returns
* Extracts and plots conditional volatility series
* Forecasts 1-step-ahead volatility and computes 1-day 95% VaR:

  * `VaR_garch = - qnorm(alpha) * sigma_forecast`
* Outputs a comparison table of VaR estimates

---

## 4. Environment Requirements

* **R**: recommended 4.1+
* **Key packages**:

  * `tidyverse`
  * `PerformanceAnalytics`
  * `moments`
  * `quadprog`
  * `gridExtra`
  * `copula`
  * `rugarch`

---

## 5. Quick Start

### 5.1 Install Dependencies

Run in R / RStudio:

```r
install.packages(c(
  "tidyverse","PerformanceAnalytics","moments","quadprog","gridExtra",
  "copula","rugarch"
))
```

### 5.2 Run the Script

1. Make sure the folder structure is in place (especially the CSV files under `data/`).
2. Run:

```bash
Rscript Portfolio_Risk_Modeling.R
```

> Note: the script currently reads `daily_price_volume.csv` from the working directory.
> If your data files are under `data/`, update file paths in the script accordingly, e.g.:

* `read.csv("data/daily_price_volume.csv")`
* `write.csv(..., file="data/daily_price_volume_returns.csv", ...)`

---

## 6. FAQ

### Q1: Why does the script fail to find the data file?

The script reads `daily_price_volume.csv` from the working directory by default. If you keep the CSVs under `data/`, update the path in `read.csv(...)` accordingly.

### Q2: Why do the GARCH plots not show real trading dates?

The script uses `as.Date(1:length(vol_gmv))` as a placeholder x-axis. If you want real dates, replace that with the aligned trading dates from the return matrix.


