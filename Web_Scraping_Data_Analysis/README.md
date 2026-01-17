# Web Scraping & Data Analysis

> **A comprehensive data pipeline and analytical framework for investigating TV show trends, longevity, and audience reception.**

## 📖 Overview

**Web Scraping & Data Analysis** is a Python-based research project designed to harvest, clean, and analyze entertainment media data. Leveraging the **TVmaze API**, this tool automates the collection of dataset specifications including air dates, ratings, and network metadata.

The project moves beyond simple data collection to perform rigorous statistical analysis, testing hypotheses regarding "Golden Age" television windows, the correlation between show longevity and ratings, and the performance gap between concluded and ongoing series.

---

## ✨ Key Features

### 🛠 Data Engineering

* **Automated Scraper:** A robust collection engine that interfaces with TVmaze to fetch canonical show data.
* **Data Serialization:** Automatically structures and exports raw data into machine-readable formats (`.csv`) for downstream analysis.
* **Attribute Extraction:** Captures critical metadata including:
* **Lifecycle:** First Air Date, End Date, Status (Running/Ended).
* **Reception:** Weighted Ratings.
* **Classification:** Genres, Network, and Summaries.



### 📊 Analytical Insights

* **Temporal Analysis (The "Golden Window"):**
* Evaluates premiere years to identify eras of peak critical reception.
* *Finding:* Statistical evidence (H=36.92) suggests the **1990s** hold a significant regression advantage over the 2010s, challenging the "modern golden age" hypothesis.


* **Status Comparative Study:**
* Compares audience reception between **Ended** and **Running** shows.
* *Finding:* Ended shows demonstrate a statistically significant higher median rating (7.90) compared to running shows (7.50).


* **Longevity vs. Quality:**
* Investigates the "Longer is Better" hypothesis using linear duration terms.
* *Finding:* The analysis reveals a non-linear "Early Rise — Mid Plateau — Late Decline" pattern, debunking the myth that longevity guarantees higher ratings.



---

## 📂 Project Structure

```text
Web_Scraping_Data_Analysis/
├── Web Scraping & Data Analysis.ipynb   # Main Jupyter Notebook (Scraping & Analysis logic)
└── README.md                            # Project Documentation

```

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.x**
* **Jupyter Notebook** or **JupyterLab**
* **Required Libraries:**
* `pandas` (Data manipulation)
* `requests` (HTTP requests)
* `scipy` (Statistical testing)
* `matplotlib` / `seaborn` (Visualization)



### Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/Project-Documentation.git

```


2. Navigate to the project directory:
```bash
cd Project-Documentation/Web_Scraping_Data_Analysis

```



### Usage Guide

1. **Launch the Notebook:**
Open `Web Scraping & Data Analysis.ipynb` in your Jupyter environment.
2. **Execute Task 1 (Scraping):**
Run the initial cells to fetch fresh data from TVmaze.
> *Note: This process will generate a local output file named `Jiahui.Hu+2252518.csv` containing the raw dataset.*


3. **Execute Task 2 (Analysis):**
Run the subsequent cells to perform statistical tests (Mann-Whitney U, Kruskal-Wallis) and generate visualizations for the Research Questions (Q1, Q2, Q3).

---

## 🚧 Roadmap & Future Enhancements

The following improvements are planned to scale the analysis and improve scraper resilience:

* **Concurrent Requests:** Implement `asyncio` or threading to speed up the scraping process for larger datasets (n > 1000).
* **Streaming Platform Integration:** Incorporate metadata from Netflix/Hulu to compare network TV vs. streaming originals.
* **Sentiment Analysis:** Apply NLP techniques to the "Summary" field to correlate plot keywords with high ratings.

---

## 🤝 Contributing

Contributions are welcome. Please follow the standard fork-and-pull request workflow:

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewAnalysis`).
3. Commit your Changes (`git commit -m 'Add NLP sentiment analysis'`).
4. Push to the Branch (`git push origin feature/NewAnalysis`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
