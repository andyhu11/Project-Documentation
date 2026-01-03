# Web Scraping & Data Analysis (TVmaze)

> A Jupyter Notebook project that **scrapes TVmaze** to collect metadata for **200 TV shows**, then performs **statistical analysis and visualization** on the exported dataset.  
> This project contains a single file: `Web Scraping & Data Analysis.ipynb`

---

## 1. Project Overview

This notebook is organized into two main parts:

- **Task 1 — Web Scraping**  
  Collects **200 show entry links** from TVmaze listing pages, then visits each show page to scrape metadata and export a CSV in a fixed schema.

- **Task 2 — Data Analysis**  
  Loads the exported CSV, performs feature engineering (e.g., decade, duration, status, network/platform, genre dummies), and applies **non-parametric tests** and **robust regression (HC3)** to study differences in ratings across decades and key covariates, along with visualizations.

**Source**: TVmaze (example seed pages include shows / popularity / rating / calendar / seasons)  
**Final CSV columns (strict order)**:  
`Title, First air date, End date, Rating, Genres, Status, Network, Summary`

---

## 2. Features

### 2.1 Task 1 — Web Scraping Pipeline
- **Requesting & parsing**
  - Uses `requests.Session()` to reuse connections + custom headers (User-Agent / Referer, etc.)
  - Parses HTML with `BeautifulSoup`
  - Basic retry logic and timeouts
  - **Random request delays** (`REQUEST_DELAY`) to reduce load

- **Entry link collection**
  - Extracts and deduplicates `/shows/<id>` show links from multiple seed listing pages
  - Automatically discovers pagination (supports both `?page=` and `/page/` patterns)
  - Outputs: `entrylinks.csv`

- **Show page metadata harvesting**
  - Extracts: Title / Summary / Rating / Genres / Status / Network
  - Aggregates air dates via the **episodes** page to compute:
    - `First air date` (earliest episode date)
    - `End date` (latest episode date)
  - Provides fallbacks for missing values (e.g., tries premiered/ended fields on the show page)
  - Exports a strict-schema CSV: `output.csv` (with assertions for row count and column order)

### 2.2 Task 2 — Data Analysis
- **Data cleaning & feature engineering**
  - Type conversions (numeric rating, parsed dates, etc.)
  - Builds key features such as `duration_years` and `decade`
  - Parses `Genres` and creates genre dummy variables (frequency-filtered)
  - Aggregates `Network` into Top-N groups (others mapped to `Other`)
  - Creates an analysis-ready dataframe (`df_an`)

- **Exploratory Data Analysis (EDA)**
  - Checks temporal coverage (whether the sample skews toward recent decades)
  - Visualizes rating distributions and trends by decade (optionally with smoothing)

- **Statistical testing (primarily non-parametric)**
  - Overall decade differences: **Kruskal–Wallis**
  - Pairwise post-hoc: **Dunn test + BH (FDR) correction**
    - Prefers `scikit-posthocs`, with a fallback implementation if unavailable
  - Status differences (e.g., Ended vs Running): **Mann–Whitney U** (with effect size reporting)

- **Regression modeling with robust inference**
  - OLS with **HC3** robust standard errors
  - Typical controls: decade fixed effects (decade dummies), duration, status, Top-N network/platform, genre dummies
  - Outputs model summaries (and optional condition number checks)
  - Optional visualizations (e.g., adjusted status effects)

---

## 3. Environment Requirements

- **Python**: 3.9+ recommended (standard scientific Python stack)
- **Core dependencies**
  - Scraping: `requests`, `beautifulsoup4`
  - Data: `pandas`, `numpy`
  - Plotting: `matplotlib`, `seaborn`
  - Stats: `scipy`, `statsmodels`
  - Optional: `scikit-posthocs` (for Dunn post-hoc tests; the notebook includes an install hint cell)

- **Runtime notes**
  - Task 1 requires internet access to reach TVmaze
  - CPU is sufficient; dataset size is small (200 rows)

---

## 4. Quick Start

### 4.1 Install dependencies (pip example)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scipy statsmodels
# Optional (recommended) for Dunn post-hoc tests
pip install scikit-posthocs
````

### 4.2 Run the notebook

```bash
jupyter notebook
```

Open and run: `Web Scraping & Data Analysis.ipynb`

### 4.3 Recommended execution order

1. **Task 1**: Run scraping and export cells first

   * Generates `entrylinks.csv`
   * Then generates `output.csv` (strictly 200 rows, 8 columns)
2. **Task 2**: Load `output.csv` and run the analysis, statistical tests, regression, and visualizations

---

## 5. Outputs & Data Format

### 5.1 Scraping outputs

* `entrylinks.csv`: collected show entry links for batch scraping
* `output.csv`: final metadata table (**strictly 200 rows & 8 columns**)

### 5.2 Final CSV schema

* `Title`: show title
* `First air date`: first air date (preferably computed from aggregated episode dates)
* `End date`: end / latest air date (preferably computed from aggregated episode dates)
* `Rating`: rating score (convertible to numeric)
* `Genres`: genre list (later parsed into dummy variables)
* `Status`: show status (e.g., Running / Ended)
* `Network`: network/platform (includes web channels where applicable)
* `Summary`: plain-text summary

> Figures are displayed inside the notebook by default (not necessarily saved to disk).

---

## 6. FAQ

### Q1: Scraping fails / timeouts happen frequently — what should I do?

* The notebook already includes delays and retries. You can increase:

  * `REQUEST_DELAY` (sleep range)
  * `TIMEOUT` (request timeout)
  * `MAX_RETRY` (number of retries)
* Avoid repeatedly rerunning Task 1; generate the CSV once, then focus on analysis.

### Q2: Why must the output be exactly 200 rows and exactly 8 columns?

The export step includes strict assertions to ensure the dataset matches the expected rubric / grading input format:

* **Row count = 200**
* **Column names and order exactly match the required schema**

### Q3: What if I don't install `scikit-posthocs`?

The notebook attempts a fallback approach for pairwise comparisons and multiple-testing correction.
However, installing `scikit-posthocs` is recommended for cleaner and more standard Dunn post-hoc testing.
