"""
Utility helper functions for data processing and formatting
"""

import pandas as pd
import numpy as np
from typing import Tuple


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero.
    
    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is zero
    
    Returns:
        Result of division with safe handling
    """
    return np.where(denominator > 0, numerator / denominator, fill_value)


def format_currency(value: float, currency: str = '€') -> str:
    """
    Format a number as currency.
    
    Args:
        value: Numeric value
        currency: Currency symbol
    
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a number as percentage.
    
    Args:
        value: Numeric value (e.g., 0.25 for 25%)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def get_date_range_from_df(df: pd.DataFrame, date_column: str = 'date') -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the min and max dates from a DataFrame.
    
    Args:
        df: DataFrame with date column
        date_column: Name of date column
    
    Returns:
        Tuple of (min_date, max_date)
    """
    return df[date_column].min(), df[date_column].max()

