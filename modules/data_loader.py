"""
Data loading and validation module
Handles CSV import and column validation
"""

import pandas as pd
import streamlit as st
from typing import List
import io


REQUIRED_COLUMNS = ['date', 'channel', 'campaign', 'spend', 'impressions', 'clicks', 'conversions', 'revenue']


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load CSV from Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        DataFrame with parsed dates and validated columns
    
    Raises:
        ValueError: If required columns missing or file cannot be parsed
    """
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        validate_columns(df)
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Check for any dates that failed to parse
        if df['date'].isna().any():
            raise ValueError(f"Some dates could not be parsed. Please check date format.")
        
        # Ensure numeric columns are numeric
        numeric_columns = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                raise ValueError(f"Column '{col}' contains non-numeric values")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty")
    except pd.errors.ParserError:
        raise ValueError("Could not parse CSV file. Please check the file format")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {str(e)}")


def validate_columns(df: pd.DataFrame) -> bool:
    """
    Check if all required columns are present.
    
    Required: date, channel, campaign, spend, impressions, 
              clicks, conversions, revenue
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}. "
            f"Required columns are: {', '.join(REQUIRED_COLUMNS)}"
        )
    
    return True


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics about the loaded data.
    
    Args:
        df: Loaded DataFrame
    
    Returns:
        Dictionary with summary statistics
    """
    return {
        'total_rows': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'channels': df['channel'].nunique(),
        'campaigns': df['campaign'].nunique(),
        'total_spend': df['spend'].sum(),
        'total_revenue': df['revenue'].sum(),
        'channel_list': sorted(df['channel'].unique().tolist()),
        'campaign_list': sorted(df['campaign'].unique().tolist())
    }

