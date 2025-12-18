"""
Data processing module
Handles metric calculations, filtering, and aggregation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from datetime import datetime


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics: ROI, CPA, CTR, Conversion Rate.
    
    Formulas:
    - ROI = revenue / spend
    - CPA = spend / conversions
    - CTR = clicks / impressions (as percentage)
    - Conversion Rate = conversions / clicks (as percentage)
    
    Note: Email CTR calculated differently (open rate × click-to-open rate)
    but we calculate the same way for consistency in visualization
    
    Args:
        df: DataFrame with raw metrics
    
    Returns:
        DataFrame with added metric columns
    """
    df = df.copy()
    
    # Calculate ROI (safe division)
    df['roi'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0)
    
    # Calculate CPA (Cost Per Acquisition)
    df['cpa'] = np.where(df['conversions'] > 0, df['spend'] / df['conversions'], 0)
    
    # Calculate CTR (Click-Through Rate) as percentage
    df['ctr'] = np.where(df['impressions'] > 0, (df['clicks'] / df['impressions']) * 100, 0)
    
    # Calculate Conversion Rate as percentage
    df['conversion_rate'] = np.where(df['clicks'] > 0, (df['conversions'] / df['clicks']) * 100, 0)
    
    # Add day of week for temporal analysis
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # Add month for seasonal analysis
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    
    return df


def filter_data(
    df: pd.DataFrame,
    start_date: datetime = None,
    end_date: datetime = None,
    channels: List[str] = None,
    campaigns: List[str] = None
) -> pd.DataFrame:
    """
    Filter data based on user selections.
    
    Args:
        df: Full dataset
        start_date: Start of date range
        end_date: End of date range
        channels: List of selected channels (None = all)
        campaigns: List of selected campaigns (None = all)
    
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Date filter
    if start_date is not None:
        filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]
    
    # Channel filter
    if channels is not None and len(channels) > 0:
        filtered_df = filtered_df[filtered_df['channel'].isin(channels)]
    
    # Campaign filter
    if campaigns is not None and len(campaigns) > 0:
        filtered_df = filtered_df[filtered_df['campaign'].isin(campaigns)]
    
    return filtered_df


def aggregate_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by channel.
    
    Args:
        df: Transaction-level data
    
    Returns:
        DataFrame with one row per channel, aggregated metrics
    """
    agg_df = df.groupby('channel').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Recalculate metrics for aggregated data
    agg_df['roi'] = np.where(agg_df['spend'] > 0, agg_df['revenue'] / agg_df['spend'], 0)
    agg_df['cpa'] = np.where(agg_df['conversions'] > 0, agg_df['spend'] / agg_df['conversions'], 0)
    agg_df['ctr'] = np.where(agg_df['impressions'] > 0, (agg_df['clicks'] / agg_df['impressions']) * 100, 0)
    agg_df['conversion_rate'] = np.where(agg_df['clicks'] > 0, (agg_df['conversions'] / agg_df['clicks']) * 100, 0)
    
    # Sort by ROI descending
    agg_df = agg_df.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return agg_df


def aggregate_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by campaign.
    
    Args:
        df: Transaction-level data
    
    Returns:
        DataFrame with one row per campaign, aggregated metrics
    """
    agg_df = df.groupby(['campaign', 'channel']).agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Recalculate metrics
    agg_df['roi'] = np.where(agg_df['spend'] > 0, agg_df['revenue'] / agg_df['spend'], 0)
    agg_df['cpa'] = np.where(agg_df['conversions'] > 0, agg_df['spend'] / agg_df['conversions'], 0)
    agg_df['ctr'] = np.where(agg_df['impressions'] > 0, (agg_df['clicks'] / agg_df['impressions']) * 100, 0)
    agg_df['conversion_rate'] = np.where(agg_df['clicks'] > 0, (agg_df['conversions'] / agg_df['clicks']) * 100, 0)
    
    # Sort by ROI descending
    agg_df = agg_df.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return agg_df


def aggregate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by date (across all channels/campaigns).
    
    Args:
        df: Transaction-level data
    
    Returns:
        DataFrame with one row per date, aggregated metrics
    """
    agg_df = df.groupby('date').agg({
        'spend': 'sum',
        'impressions': 'sum',
        'clicks': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Recalculate metrics
    agg_df['roi'] = np.where(agg_df['spend'] > 0, agg_df['revenue'] / agg_df['spend'], 0)
    agg_df['cpa'] = np.where(agg_df['conversions'] > 0, agg_df['spend'] / agg_df['conversions'], 0)
    agg_df['ctr'] = np.where(agg_df['impressions'] > 0, (agg_df['clicks'] / agg_df['impressions']) * 100, 0)
    agg_df['conversion_rate'] = np.where(agg_df['clicks'] > 0, (agg_df['conversions'] / agg_df['clicks']) * 100, 0)
    
    return agg_df


def get_active_campaigns_for_date_range(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> List[str]:
    """
    Get list of campaigns that were active during the specified date range.
    
    Args:
        df: Full dataset
        start_date: Start of date range
        end_date: End of date range
    
    Returns:
        List of campaign names that had activity in the date range
    """
    filtered = filter_data(df, start_date=start_date, end_date=end_date)
    return sorted(filtered['campaign'].unique().tolist())

