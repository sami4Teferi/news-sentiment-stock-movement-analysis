# Correlation.py
import pandas as pd
from textblob import TextBlob
from scipy.stats import pearsonr
from typing import Dict  # Add this import to resolve the NameError

def align_dates(stock_dfs, news_df, start_date='2023-01-01', end_date='2024-12-31'):
    """
    Align stock and news datasets by date, filtering to the specified range.
    
    Args:
        stock_dfs (dict): Dictionary of stock DataFrames with Date as index.
        news_df (pd.DataFrame): News DataFrame with Date column.
        start_date (str): Start date for filtering (inclusive).
        end_date (str): End date for filtering (inclusive).
    
    Returns:
        tuple: (stock_dfs, news_df) with aligned dates.
    """
    # Convert start_date and end_date to UTC-aware datetime
    start_date = pd.to_datetime(start_date, utc=True)
    end_date = pd.to_datetime(end_date, utc=True)
    
    # Filter stock data to the specified date range
    for ticker, df in stock_dfs.items():
        df.index = pd.to_datetime(df.index, utc=True)  # Ensure index is UTC-aware
        stock_dfs[ticker] = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Normalize news dates (ensure UTC-aware and remove time component)
    news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True).dt.normalize()
    
    # Filter news to the specified date range
    news_df = news_df[(news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)]
    
    return stock_dfs, news_df

def perform_sentiment_analysis(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis on news headlines using TextBlob.
    
    Args:
        news_df (pd.DataFrame): News DataFrame with a 'Headline' column.
    
    Returns:
        pd.DataFrame: News DataFrame with an additional 'Sentiment' column.
    """
    news_df['Sentiment'] = news_df['Headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return news_df

def aggregate_sentiments(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date and ticker, computing the mean sentiment.
    
    Args:
        news_df (pd.DataFrame): News DataFrame with 'Date', 'Ticker', and 'Sentiment' columns.
    
    Returns:
        pd.DataFrame: Pivot table with dates as index, tickers as columns, and mean sentiment as values.
    """
    daily_sentiment = news_df.groupby(['Date', 'Ticker'])['Sentiment'].mean().reset_index()
    daily_sentiment_pivot = daily_sentiment.pivot(index='Date', columns='Ticker', values='Sentiment')
    
    return daily_sentiment_pivot

def calculate_correlation(stock_dfs: Dict[str, pd.DataFrame], daily_sentiment_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation between daily sentiment scores and stock returns.
    
    Args:
        stock_dfs (Dict[str, pd.DataFrame]): Dictionary of stock DataFrames.
        daily_sentiment_pivot (pd.DataFrame): Pivot table of daily sentiment scores.
    
    Returns:
        pd.DataFrame: DataFrame with correlation and p-value for each ticker.
    """
    correlations = {}
    
    for ticker in stock_dfs.keys():
        if ticker in daily_sentiment_pivot.columns:
            # Merge stock returns and sentiment scores on date
            stock_df = stock_dfs[ticker][['Daily_Return']].copy()
            sentiment_series = daily_sentiment_pivot[ticker].dropna()
            merged_df = stock_df.join(sentiment_series.rename(ticker + '_Sentiment'), how='inner')
            
            if len(merged_df) > 1:  # Need at least 2 data points for correlation
                corr, p_value = pearsonr(merged_df['Daily_Return'], merged_df[ticker + '_Sentiment'])
                correlations[ticker] = {'Correlation': corr, 'P-Value': p_value}
    
    return pd.DataFrame(correlations).T