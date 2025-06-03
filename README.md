# News Sentiment and Stock Movement Analysis

This project analyzes the relationship between financial news sentiment and stock price movements to support predictive analytics for Nova Financial Insights. By combining natural language processing (NLP) based sentiment analysis with quantitative finance techniques, it aims to uncover actionable insights for forecasting stock price trends.

---

## Project Overview

- **Objective:**  
  Investigate how news sentiment correlates with subsequent stock returns to improve forecasting accuracy.

- **Approach:**  
  - Download and process historical stock prices using `yfinance`.  
  - Compute technical indicators like Moving Averages, RSI, MACD using `TA-Lib`.  
  - Perform sentiment analysis on financial news headlines with 

---


---

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/sami4Teferi/news-sentiment-stock-analysis.git
    cd news-sentiment-stock-analysis
    ```

2. Create and activate a Python virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download or place historical stock price CSV files in `data/yfinance_data/`.  
   Ensure financial news headline data is available in `data/financial_news.csv`.

---

## Usage

- **Run the Jupyter notebooks** for step-by-step analysis and visualization:
    - `notebooks/Quantitative_analysis.ipynb` — technical indicators and stock trends  
    - `notebooks/Sentiment_analysis.ipynb` — sentiment scoring and correlation analysis  

- **Modify ticker symbols, date ranges, and other parameters** within notebooks or scripts as needed.

---

## Key Technologies

- [Python 3.8+](https://www.python.org/downloads/)
- [yfinance](https://pypi.org/project/yfinance/) — Yahoo Finance API for historical stock data
- [TA-Lib](https://mrjbq7.github.io/ta-lib/) — Technical analysis indicators (SMA, RSI, MACD)
- [TextBlob](https://textblob.readthedocs.io/) — NLP library for sentiment analysis
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) — Data processing
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) — Visualization

---

## Findings

- Positive financial news sentiment tends to align with upward stock price movements, while negative sentiment often precedes declines.  
- The Pearson correlation coefficient between news sentiment polarity and next-day stock returns is positive, suggesting sentiment can be a useful early indicator for market trends.  
- Combining sentiment analysis with traditional technical indicators can enhance predictive models for stock price forecasting.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---







