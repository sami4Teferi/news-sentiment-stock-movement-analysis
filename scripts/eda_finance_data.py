import pandas as pd
import os

def load_financial_data(file_paths):
    """
    Load multiple CSVs into a dictionary of DataFrames.

    Args:
        file_paths (dict): Dictionary with keys as stock symbols and values as file paths.
    
    Returns:
        dict: Dictionary of DataFrames
    """
    dataframes = {}
    for symbol, path in file_paths.items():
        df = pd.read_csv(path)
        dataframes[symbol] = df
    return dataframes

def basic_eda(dataframes, verbose=True):
    """
    Perform basic EDA on each DataFrame.

    Args:
        dataframes (dict): Dictionary of stock DataFrames.
        verbose (bool): Whether to print EDA results.
    
    Returns:
        dict: Dictionary of EDA summaries for each stock.
    """
    summary = {}

    for symbol, df in dataframes.items():
        info = {
            'Shape': df.shape,
            'Missing Values': df.isnull().sum(),
            'Duplicated Rows': df.duplicated().sum(),
            'Describe': df.describe(include='all')
        }
        summary[symbol] = info

        if verbose:
            print(f"\n================================== {symbol} ===========================================")
            print("Shape:", info['Shape'])
            print("Missing Values:\n", info['Missing Values'])
            print("Duplicated Rows:", info['Duplicated Rows'])
            print("Summary Stats:\n", info['Describe'])
    
    return summary
