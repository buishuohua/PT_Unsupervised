# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Configuration for Alpha Vantage API
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlphaVantageConfig:
    """Configuration class for Alpha Vantage API parameters"""

    # API credentials and base URL
    api_key: str = "HF3NKWYHVA7FI9GR"  # Replace with your actual API key
    base_url: str = "https://www.alphavantage.co/query"

    # List of symbols to fetch
    symbols: List[str] = field(default_factory=list)

    # API parameters
    function: str = "TIME_SERIES_INTRADAY"
    interval: str = "1min"
    adjusted: bool = True
    extended_hours: bool = False

    def add_symbols(self, *symbols: str) -> None:
        """Add symbols to the list"""
        self.symbols.extend(symbols)

    def to_dict(self, symbol: Optional[str] = None) -> dict:
        """Convert config to dictionary format for API request"""
        params = {
            "apikey": self.api_key,
            "function": self.function,
            "interval": self.interval,
            "outputsize": "full",
            "adjusted": str(self.adjusted).lower(),
            "extended_hours": str(self.extended_hours).lower()
        }

        if symbol:
            params["symbol"] = symbol

        return params
