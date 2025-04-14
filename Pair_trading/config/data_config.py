# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Configuration for data
"""


from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """Configuration class for data parameters"""

    # Data parameters
    data_dir: str = "data"
    processed_data_dir: str = "processed"
    interval: str = "1min"

    # Time periods
    train_begin_year: int = 2016
    train_end_year: int = 2020
    val_begin_year: int = 2021
    val_end_year: int = 2022
    test_begin_year: int = 2023
    test_end_year: int = 2025

    # Sequence parameters
    seq_length: int = 20
    pred_length: int = 2

    # Trading pairs configuration
    pairs: Tuple[Tuple[str, ...], ...] = (
        ('AAPL', 'MSFT'),
        ('GOOGL', 'META', 'AMZN')
    )
    hedge_ratios: Tuple[Tuple[float, ...], ...] = (
        (1.0,),
        (1.0, 1.0)
    )

    # Technical indicators
    ema_window_size: int = 10

    # Data processing
    feature_columns: Tuple[str, ...] = (
        'open', 'high', 'low', 'close', 'volume')
