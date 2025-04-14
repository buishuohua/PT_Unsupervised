# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Fetch data from Alpha Vantage API
"""

from datetime import datetime
from typing import Dict
import pandas as pd
import requests
import time
import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from Pair_trading.config.AlphaV_config import AlphaVantageConfig


class AlphaVantage:
    """Class to handle Alpha Vantage API requests"""

    def __init__(self):
        self.session = requests.Session()
        self.rate_limit_delay = 0

    def fetch_extended_intraday(self, config: AlphaVantageConfig, years: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Fetch extended intraday data for multiple symbols using TIME_SERIES_INTRADAY

        Args:
            config: AlphaVantageConfig object
            years: Number of years of data to fetch (default: 10)

        Returns:
            Dict of DataFrames containing minute-based data for each symbol
        """
        if not config.symbols:
            print("No symbols configured. Use config.add_symbols() to add symbols.")
            return {}

        all_data = {}
        time_series_key = f"Time Series ({config.interval})"

        data_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        for symbol in config.symbols:
            print(f"\nFetching data for {symbol}...")
            symbol_data = []

            current_year = datetime.now().year
            current_month = datetime.now().month

            start_year = max(2000, current_year - years + 1)

            raw_data_dir = os.path.join(data_dir, "raw")
            temp_dir = os.path.join(raw_data_dir, f"temp_{symbol}")
            os.makedirs(temp_dir, exist_ok=True)

            for year in range(start_year, current_year + 1):
                start_month = 1
                end_month = 12 if year < current_year else current_month

                for month in range(start_month, end_month + 1):
                    month_file = os.path.join(
                        temp_dir, f"{symbol}_{year}_{month:02d}.csv")

                    if os.path.exists(month_file) and os.path.getsize(month_file) > 0:
                        print(
                            f"Loading existing data for {symbol} - {year}/{month:02d}")
                        month_df = pd.read_csv(month_file)
                        month_df['timestamp'] = pd.to_datetime(
                            month_df['timestamp'])
                        symbol_data.append(month_df)
                        continue

                    try:
                        params = {
                            "function": "TIME_SERIES_INTRADAY",
                            "symbol": symbol,
                            "interval": config.interval,
                            "month": f"{year}-{month:02d}",
                            "outputsize": "full",
                            "adjusted": "true",
                            "extended_hours": "true",
                            "apikey": config.api_key,
                            "datatype": "json"
                        }

                        print(
                            f"Fetching data for {symbol} - {year}/{month:02d}")

                        response = self.session.get(
                            config.base_url, params=params)
                        response.raise_for_status()
                        data = response.json()

                        if "Error Message" in data:
                            print(
                                f"API Error for {symbol}: {data['Error Message']}")
                            continue

                        if "Note" in data:
                            print(f"API Note for {symbol}: {data['Note']}")
                            continue

                        time_series_data = data.get(time_series_key)
                        if not time_series_data:
                            print(
                                f"No data found for {symbol} - {year}/{month:02d}")
                            continue

                        df = pd.DataFrame.from_dict(
                            time_series_data, orient='index')
                        df.columns = [col.split('. ')[1] for col in df.columns]

                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        df.index = pd.to_datetime(df.index)
                        df['timestamp'] = df.index
                        df['ticker'] = symbol
                        df = df.reset_index(drop=True)

                        cols = ['timestamp', 'ticker', 'open',
                                'high', 'low', 'close', 'volume']
                        df = df[cols]

                        df.to_csv(month_file, index=False)
                        symbol_data.append(df)

                        print(
                            f"Saved {len(df)} records for {year}/{month:02d}")

                        time.sleep(self.rate_limit_delay)

                    except Exception as e:
                        print(
                            f"Error processing {symbol} {year}/{month:02d}: {str(e)}")

            if symbol_data:
                combined_df = pd.concat(symbol_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')

                all_data[symbol] = combined_df

                os.makedirs(raw_data_dir, exist_ok=True)

                symbol_file = os.path.join(
                    raw_data_dir, f"{symbol}_{config.interval}.csv")
                combined_df.to_csv(symbol_file, index=False)

                print(f"\nTotal records for {symbol}: {len(combined_df)}")
                print(
                    f"Complete date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

                shutil.rmtree(temp_dir)

            if len(all_data) == len(config.symbols):
                all_symbols_df = pd.concat(
                    all_data.values(), ignore_index=True)
                all_symbols_df = all_symbols_df.sort_values(
                    ['timestamp', 'ticker'])
                all_symbols_file = os.path.join(
                    raw_data_dir, f"All_{config.interval}_data.csv")
                all_symbols_df.to_csv(all_symbols_file, index=False)
                print(
                    f"\nSaved combined data for all symbols to {all_symbols_file}")
                print(f"Total combined records: {len(all_symbols_df)}")

        return all_data


if __name__ == "__main__":
    config = AlphaVantageConfig()
    tickers = ("AAPL", "MSFT", "GOOGL", "META", "AMD", "NVDA", "INTC", "CSCO", "ORCL", "IBM", "JPM", "BAC", "C", "WFC", "GS", "MS", "V", "MA", "AXP", "COF", "XOM", "CVX", "COP", "MRO",
               "SLB", "HAL", "JNJ", "PFE", "MRK", "BMY", "UNH", "HUM", "CVS", "WBA", "WMT", "TGT", "HD", "LOW", "COST", "KR", "T", "VZ", "TMUS", "F", "GM", "TSLA", "DAL", "AAL", "UAL", "LUV")
    config.add_symbols(*tickers)
    config.interval = "1min"
    av = AlphaVantage()
    data = av.fetch_extended_intraday(config, years=10)
