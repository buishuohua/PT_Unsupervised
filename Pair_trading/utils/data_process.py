# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Data process
"""

import pandas as pd
import numpy as np
from typing import Dict
import os


def align_stock_timestamps(data_dir: str, interval: str = "1min", begin_year: int = None, end_year: int = None) -> pd.DataFrame:
    """
    Align stock data from existing combined data file to ensure all stocks
    have the same timestamps within specified year range.

    Args:
        data_dir: Directory containing the data files
        interval: Time interval of the data (default: "1min")
        begin_year: Start year for filtering data (inclusive)
        end_year: End year for filtering data (inclusive)

    Returns:
        DataFrame with aligned data for all stocks
    """
    raw_data_dir = os.path.join(data_dir, "raw")
    input_file = os.path.join(raw_data_dir, f"All_{interval}_data.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Data file not found: {input_file}")

    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter by year range if specified
    if begin_year is not None:
        df = df[df['timestamp'].dt.year >= begin_year]
    if end_year is not None:
        df = df[df['timestamp'].dt.year <= end_year]

    # Get counts per timestamp for each ticker
    timestamp_counts = df.groupby('timestamp')['ticker'].nunique()

    # Find timestamps that have data for all tickers
    n_tickers = df['ticker'].nunique()
    complete_timestamps = timestamp_counts[timestamp_counts == n_tickers].index

    # Filter data to keep only complete timestamps
    aligned_df = df[df['timestamp'].isin(complete_timestamps)].copy()
    aligned_df = aligned_df.sort_values(['timestamp', 'ticker'])

    # Verify alignment
    ticker_counts = aligned_df.groupby('ticker').size()

    # Save aligned data
    year_suffix = f"_{begin_year}_{end_year}" if begin_year and end_year else ""
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_file = os.path.join(
        processed_dir, f"Aligned_{interval}{year_suffix}_data.csv")
    aligned_df.to_csv(output_file, index=False)

    print(f"\nAlignment Summary:")
    print(f"Original timestamps: {len(df['timestamp'].unique())}")
    print(f"Aligned timestamps: {len(complete_timestamps)}")
    print(f"Number of tickers: {n_tickers}")
    print(f"Records per ticker: {len(complete_timestamps)}")
    print(f"Total records: {len(aligned_df)}")
    print(
        f"Date range: {aligned_df['timestamp'].min()} to {aligned_df['timestamp'].max()}")
    print(f"Saved aligned data to: {output_file}")

    if len(set(ticker_counts)) != 1:
        print("\nWarning: Uneven number of records per ticker:")
        print(ticker_counts)

    return aligned_df


def spread_pairs(df: pd.DataFrame, stocks: tuple, hedge_ratios: tuple = None) -> tuple:
    """
    Calculate spread between multiple stocks and ensure index alignment.

    Args:
        df: DataFrame containing stock data
        stocks: Tuple of stock tickers (e.g., ('AAPL', 'MSFT', 'GOOGL'))
        hedge_ratios: Tuple of hedge ratios for each stock after the first one (default: None, all 1.0)
    Returns:
        tuple: (list_of_stock_dfs, spread_series) with aligned indices
    """
    if hedge_ratios is None:
        hedge_ratios = tuple([1.0] * (len(stocks) - 1))

    if len(hedge_ratios) != len(stocks) - 1:
        raise ValueError(
            "Number of hedge ratios must be equal to number of stocks minus 1")

    # Get all stock DataFrames and find common timestamps
    stock_dfs = []
    common_index = None

    # First, get all stock DataFrames and find common timestamps
    for stock in stocks:
        stock_df = df[df["ticker"] == stock].copy()
        stock_df.set_index('timestamp', inplace=True)
        stock_dfs.append(stock_df)

        if common_index is None:
            common_index = set(stock_df.index)
        else:
            common_index = common_index.intersection(set(stock_df.index))

    common_index = sorted(list(common_index))

    # Filter all DataFrames to include only common timestamps
    filtered_stock_dfs = []
    for stock_df in stock_dfs:
        filtered_df = stock_df.loc[common_index].copy()
        filtered_stock_dfs.append(filtered_df)

    # Calculate spread with aligned indices
    spread = pd.Series(
        filtered_stock_dfs[0]["close"].copy(),  # First stock as base
        index=common_index,
        name='spread'
    )

    # Subtract weighted prices of other stocks
    for stock_df, hedge_ratio in zip(filtered_stock_dfs[1:], hedge_ratios):
        spread -= hedge_ratio * stock_df["close"]

    # Verify alignment
    lengths = [len(df) for df in filtered_stock_dfs]
    indices = [df.index for df in filtered_stock_dfs]

    # Verify all lengths are equal
    assert len(set(lengths)) == 1, "Length mismatch in spread calculation"

    # Verify all indices match
    for idx in indices:
        assert all(idx == indices[0]), "Index mismatch between stocks"

    assert all(spread.index == indices[0]
               ), "Index mismatch between spread and stocks"

    return filtered_stock_dfs, spread


def ema(spread: pd.Series, window_size: int = 2) -> pd.Series:
    """
    Calculate EMA while preserving the original index.
    
    Args:
        spread: Series containing spread values with timestamp index
        window_size: Window size for EMA calculation (default: 1)
    
    Returns:
        Series: EMA values with same index as input spread
    """
    ema_series = pd.Series(
        spread.ewm(span=window_size).mean(),
        index=spread.index,
        name='ema'
    )

    # Verify index alignment
    assert all(ema_series.index ==
               spread.index), "Index mismatch between EMA and spread"

    return ema_series


def create_sequences(stock_dfs: list, spread: pd.Series,
                     seq_length: int = 3, pred_length: int = 2) -> tuple:
    """
    Create sequences of data for time series prediction.
    
    Args:
        stock_dfs: List of DataFrames for each stock
        spread: Series containing spread values
        seq_length: Length of input sequence (default: 10)
        pred_length: Length of prediction window (default: 5)
    
    Returns:
        tuple: (stock_sequences, spread_sequences, labels) where:
            - stock_sequences: list of (num_stocks, seq_length, features) arrays
            - spread_sequences: list of spread sequences
            - labels: list of labels
    """
    total_length = len(spread)
    stock_sequences = []
    spread_sequences = []
    labels = []

    features_columns = ['open', 'high', 'low', 'close', 'volume']

    for i in range(total_length - seq_length - pred_length + 1):
        # Get sequences for all stocks
        stock_seqs = []
        for stock_df in stock_dfs:
            # (seq_length, features)
            seq = stock_df.iloc[i:i+seq_length][features_columns].values
            stock_seqs.append(seq)

        seq_spread = spread.iloc[i:i+seq_length]

        # Stack sequences for all stocks: (num_stocks, seq_length, features)
        stocks_seq = np.stack(stock_seqs)

        # Get future spread values for labeling
        future_spread = spread.iloc[i+seq_length:i+seq_length+pred_length]

        # Calculate label based on future trend
        current_mean = seq_spread.mean()
        current_std = seq_spread.std()
        future_mean = future_spread.mean()

        # Create label based on thresholds
        if future_mean > current_mean + 0.2 * current_std:
            label = 2  # Upward trend
        elif future_mean < current_mean - 0.2 * current_std:
            label = 0  # Downward trend
        else:
            label = 1  # Sideways

        stock_sequences.append(stocks_seq)
        spread_sequences.append(seq_spread)
        labels.append(label)

    return stock_sequences, spread_sequences, labels


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    aligned_data = align_stock_timestamps(data_dir, interval="1min", begin_year=2016, end_year=2020)
    aligned_data2 = align_stock_timestamps(data_dir, interval="1min", begin_year=2021, end_year=2022)
    aligned_data3 = align_stock_timestamps(data_dir, interval="1min", begin_year=2023, end_year=2025)
