import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import pynance as pn
import yfinance as yf
import numpy as np
import os
from IPython.display import display



sns.set(style="darkgrid")

def load_stock_data(filepath=None, ticker=None, start_date=None, end_date=None):
    """
    Load stock price data from a CSV file or fetch using yfinance/PyNance.
    Validates required columns and data integrity.
    """
    try:
        if filepath:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            df = pd.read_csv(filepath)
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isna().any():
                raise ValueError("Invalid or missing dates in 'Date' column")
            
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
        
        elif ticker and start_date and end_date:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    raise ValueError(f"No data fetched for {ticker}")
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Data for {ticker} missing required columns: {required_cols}")
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
            except Exception as e:
                print(f"yfinance failed for {ticker}: {e}. Trying PyNance...")
                df = pn.data.get(ticker, start=start_date, end=end_date)
                if df.empty:
                    raise ValueError(f"No data fetched for {ticker}")
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Data for {ticker} missing required columns: {required_cols}")
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
        else:
            raise ValueError("Provide either filepath or ticker with start/end dates")
        
        if len(df) < 50:
            raise ValueError("Insufficient data points for technical indicators (minimum 50 required)")
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def add_technical_indicators(df, drop_na=False):
    """
    Apply TA-Lib technical indicators: SMA, RSI, MACD, Bollinger Bands.
    Optionally drop rows with NaN indicator values.
    """
    try:
        df = df.copy()
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['MACD_hist'] = macdhist
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)
        if drop_na:
            df = df.dropna(subset=['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper'])
        return df
    except Exception as e:
        print(f"Error adding technical indicators: {e}")
        return df

def add_trading_signals(df):
    """
    Add buy/sell signals based on SMA crossovers and RSI thresholds.
    """
    try:
        df = df.copy()
        df['Signal'] = 0
        df.loc[df['SMA_20'] > df['SMA_50'], 'Signal'] = 1  # Buy
        df.loc[df['SMA_20'] < df['SMA_50'], 'Signal'] = -1  # Sell
        df.loc[df['RSI'] < 30, 'Signal'] = 1  # Buy (oversold)
        df.loc[df['RSI'] > 70, 'Signal'] = -1  # Sell (overbought)
        return df
    except Exception as e:
        print(f"Error adding trading signals: {e}")
        return df

def calculate_financial_metrics(df):
    """
    Calculate financial metrics: daily returns, volatility, Sharpe ratio.
    """
    try:
        df = df.copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)
        df['Sharpe_Ratio'] = (df['Daily_Return'].rolling(window=21).mean() * 252) / df['Volatility']
        return df
    except Exception as e:
        print(f"Error calculating financial metrics: {e}")
        return df

def backtest_signals(df):
    """
    Backtest trading signals and calculate strategy returns.
    Returns cumulative return and number of trades.
    """
    try:
        df = df.copy()
        df['Position'] = df['Signal'].shift(1)  # Hold position after signal
        df['Strategy_Return'] = df['Daily_Return'] * df['Position']
        cumulative_return = (1 + df['Strategy_Return'].fillna(0)).cumprod().iloc[-1] - 1
        num_trades = df['Signal'].diff().abs().sum() / 2  # Count signal changes
        return cumulative_return, num_trades
    except Exception as e:
        print(f"Error backtesting signals: {e}")
        return 0.0, 0

def print_statistical_summary(df, stock_name):
    """
    Print statistical summary of stock data and indicators.
    """
    try:
        print(f"\nStatistical Summary for {stock_name}")
        print("-" * 50)
        print("Price Statistics:")
        display(df[['Close', 'Daily_Return', 'Volatility', 'Sharpe_Ratio']].describe())
        print("\nIndicator Statistics:")
        display(df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].describe())
        print("\nCorrelation Matrix:")
        display(df[['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Daily_Return']].corr())
        cum_return, num_trades = backtest_signals(df)
        print(f"\nTrading Strategy Performance:")
        print(f"Cumulative Return: {cum_return:.2%}")
        print(f"Number of Trades: {int(num_trades)}")
    except Exception as e:
        print(f"Error printing statistical summary: {e}")

def plot_price_with_indicators(df, stock_name, save_path=None):
    """
    Plot price, SMAs, volume, RSI, MACD, Bollinger Bands, and trading signals.
    Optionally save plot to file.
    """
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Price, SMAs, Bollinger Bands, and signals
        ax1.plot(df['Close'], label='Close Price', color='blue')
        ax1.plot(df['SMA_20'], label='SMA 20', color='orange')
        ax1.plot(df['SMA_50'], label='SMA 50', color='green')
        ax1.plot(df['BB_upper'], label='BB Upper', color='purple', linestyle='--')
        ax1.plot(df['BB_lower'], label='BB Lower', color='purple', linestyle='--')
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
        ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)
        ax1.set_title(f"{stock_name} Price, SMAs, and Bollinger Bands")
        ax1.legend()
        
        # Plot 2: Volume
        ax2.bar(df.index, df['Volume'], color='grey', alpha=0.5)
        ax2.set_title(f"{stock_name} Volume")
        
        # Plot 3: RSI and MACD
        ax3.plot(df['RSI'], label='RSI', color='purple')
        ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax3.set_title(f"{stock_name} RSI and MACD")
        ax3.legend(loc='upper left')
        
        ax3b = ax3.twinx()
        ax3b.plot(df['MACD'], label='MACD', color='blue', alpha=0.5)
        ax3b.plot(df['MACD_signal'], label='Signal Line', color='orange', alpha=0.5)
        ax3b.bar(df.index, df['MACD_hist'], label='MACD Hist', color='grey', alpha=0.3)
        ax3b.legend(loc='upper right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        plt.show()
    except Exception as e:
        print(f"Error plotting indicators: {e}")