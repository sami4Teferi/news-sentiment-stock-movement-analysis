import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Ensure plots look nice
sns.set(style="darkgrid")

# ----------------------------
# 1. Load and Prepare the Data
# ----------------------------
def load_stock_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# ----------------------------------
# 2. Apply Technical Indicators (TA-Lib)
# ----------------------------------
def add_technical_indicators(df):
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    df['MACD_hist'] = macdhist
    return df

# ----------------------------------
# 3. Visualization Functions
# ----------------------------------
def plot_price_with_indicators(df, stock_name):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['SMA_20'], label='SMA 20')
    plt.plot(df['SMA_50'], label='SMA 50')
    plt.title(f"{stock_name} Price with SMA Indicators")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title(f"{stock_name} RSI")
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['MACD_signal'], label='Signal Line', color='orange')
    plt.bar(df.index, df['MACD_hist'], label='MACD Hist', color='grey')
    plt.title(f"{stock_name} MACD")
    plt.legend()
    plt.show()

# Optional Placeholder for PyNance
# def use_pynance():
#     import pynance as pn
#     # Example: pn.get(name="AAPL", start="2023-01-01", end="2023-12-31")
#     pass

# ----------------------
# Wrapper for full workflow
# ----------------------
def full_quantitative_analysis(filepath, stock_name):
    df = load_stock_data(filepath)
    df = add_technical_indicators(df)
    plot_price_with_indicators(df, stock_name)
    return df
