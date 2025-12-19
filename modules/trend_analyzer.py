"""
Trend analysis module
Calculates trends using moving averages and linear regression
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.linear_model import LinearRegression


def analyze_trends(
    df: pd.DataFrame,
    metric: str = 'conversions',
    window: int = 7
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate moving average and fit linear trend.
    
    Args:
        df: DataFrame sorted by date
        metric: Column name to analyze
        window: Days for moving average
    
    Returns:
        Tuple of (df_with_trends, trend_stats)
        
        df_with_trends has added columns:
        - {metric}_ma{window}: moving average
        - {metric}_trend: linear trend line
        
        trend_stats = {
            'direction': 'increasing' or 'decreasing',
            'slope': float,
            'percent_change': float,
            'r_squared': float,
            'strength': 'strong' | 'moderate' | 'weak'
        }
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate moving average
    ma_col = f'{metric}_ma{window}'
    df[ma_col] = df[metric].rolling(window=window, min_periods=1).mean()
    
    # Prepare data for linear regression
    if len(df) < 2:
        # Not enough data for trend analysis
        df[f'{metric}_trend'] = df[metric]
        return df, {
            'direction': 'unknown',
            'slope': 0,
            'percent_change': 0,
            'r_squared': 0,
            'strength': 'weak'
        }
    
    # Create time index (days from start)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions (trend line)
    df[f'{metric}_trend'] = model.predict(X)
    
    # Calculate R²
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate slope and percent change
    slope = model.coef_[0]
    start_value = df[metric].iloc[:min(7, len(df))].mean()
    end_value = df[metric].iloc[-min(7, len(df)):].mean()
    percent_change = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
    
    # Determine direction
    direction = 'increasing' if slope > 0 else 'decreasing'
    
    # Classify strength based on R²
    if r_squared > 0.7:
        strength = 'strong'
    elif r_squared > 0.4:
        strength = 'moderate'
    else:
        strength = 'weak'
    
    trend_stats = {
        'direction': direction,
        'slope': float(slope),
        'percent_change': float(percent_change),
        'r_squared': float(r_squared),
        'strength': strength,
        'start_value': float(start_value),
        'end_value': float(end_value)
    }
    
    return df, trend_stats


def analyze_multiple_metrics(
    df: pd.DataFrame,
    metrics: list = ['conversions', 'revenue', 'spend'],
    window: int = 7
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Analyze trends for multiple metrics.
    
    Args:
        df: DataFrame with data
        metrics: List of metric columns to analyze
        window: Moving average window
    
    Returns:
        Tuple of (df_with_all_trends, dict_of_trend_stats_by_metric)
    """
    all_stats = {}
    result_df = df.copy()
    
    for metric in metrics:
        if metric in df.columns:
            result_df, stats = analyze_trends(result_df, metric=metric, window=window)
            all_stats[metric] = stats
    
    return result_df, all_stats


def detect_day_of_week_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze performance by day of week.
    
    Args:
        df: DataFrame with day_of_week column and metrics
    
    Returns:
        Dictionary with day-of-week insights
    """
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['date'].dt.day_name()
    
    # Aggregate by day of week
    dow_analysis = df.groupby('day_of_week').agg({
        'conversions': 'mean',
        'revenue': 'mean',
        'spend': 'mean'
    })
    
    # Calculate weekend effect
    weekend_days = ['Saturday', 'Sunday']
    weekday_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    weekend_conversions = df[df['day_of_week'].isin(weekend_days)]['conversions'].mean()
    weekday_conversions = df[df['day_of_week'].isin(weekday_days)]['conversions'].mean()
    
    weekend_effect = ((weekend_conversions - weekday_conversions) / weekday_conversions * 100) if weekday_conversions > 0 else 0
    
    # Find best and worst days
    best_day = dow_analysis['conversions'].idxmax()
    worst_day = dow_analysis['conversions'].idxmin()
    
    insights = []
    if weekend_effect < -20:
        insights.append(f"Conversions are {abs(weekend_effect):.0f}% lower on weekends")
    elif weekend_effect > 20:
        insights.append(f"Conversions are {weekend_effect:.0f}% higher on weekends")
    
    return {
        'weekend_effect': float(weekend_effect),
        'best_day': best_day,
        'worst_day': worst_day,
        'insights': insights,
        'by_day': dow_analysis.to_dict('index')
    }


def detect_seasonal_patterns(df: pd.DataFrame) -> list:
    """
    Identify seasonal trends (New Year, Valentine's, month-end).
    
    Args:
        df: DataFrame with date column
    
    Returns:
        List of seasonal insight strings
    """
    insights = []
    
    # Check for New Year effect (first 15 days of January)
    jan_data = df[(df['date'].dt.month == 1) & (df['date'].dt.day <= 15)]
    if len(jan_data) > 0:
        jan_avg_conversions = jan_data['conversions'].mean()
        other_avg_conversions = df[~((df['date'].dt.month == 1) & (df['date'].dt.day <= 15))]['conversions'].mean()
        
        if other_avg_conversions > 0:
            jan_effect = ((jan_avg_conversions - other_avg_conversions) / other_avg_conversions * 100)
            if abs(jan_effect) > 15:
                insights.append(f"New Year period (Jan 1-15) shows {abs(jan_effect):.0f}% {'higher' if jan_effect > 0 else 'lower'} conversions")
    
    # Check for Valentine's Day effect (Feb 10-14)
    valentine_data = df[(df['date'].dt.month == 2) & (df['date'].dt.day >= 10) & (df['date'].dt.day <= 14)]
    if len(valentine_data) > 0:
        valentine_avg = valentine_data['conversions'].mean()
        feb_other_avg = df[(df['date'].dt.month == 2) & ((df['date'].dt.day < 10) | (df['date'].dt.day > 14))]['conversions'].mean()
        
        if feb_other_avg > 0:
            valentine_effect = ((valentine_avg - feb_other_avg) / feb_other_avg * 100)
            if abs(valentine_effect) > 15:
                insights.append(f"Valentine's period (Feb 10-14) shows {abs(valentine_effect):.0f}% {'higher' if valentine_effect > 0 else 'lower'} conversions")
    
    # Check for month-end effect (last 3 days of month)
    df['is_month_end'] = df['date'].dt.day >= df['date'].dt.days_in_month - 2
    month_end_avg = df[df['is_month_end']]['spend'].mean()
    month_other_avg = df[~df['is_month_end']]['spend'].mean()
    
    if month_other_avg > 0:
        month_end_effect = ((month_end_avg - month_other_avg) / month_other_avg * 100)
        if abs(month_end_effect) > 10:
            insights.append(f"Month-end spend is {abs(month_end_effect):.0f}% {'higher' if month_end_effect > 0 else 'lower'} (budget pacing)")
    
    return insights


def forecast_metric(
    df: pd.DataFrame,
    metric: str = 'conversions',
    days_ahead: int = 7
) -> Tuple[pd.DataFrame, Dict]:
    """
    Forecast future metric values using linear regression (ML-based prediction).
    
    Uses the trend from historical data to predict future values.
    
    Args:
        df: DataFrame sorted by date with metric column
        metric: Column name to forecast
        days_ahead: Number of days to forecast ahead
    
    Returns:
        Tuple of (forecast_df, forecast_stats)
        
        forecast_df contains:
        - date: Future dates
        - {metric}_forecast: Predicted values
        - {metric}_forecast_lower: Lower confidence bound
        - {metric}_forecast_upper: Upper confidence bound
        
        forecast_stats contains:
        - next_value: Predicted value for next day
        - trend_direction: 'increasing' or 'decreasing'
        - confidence: 'high' | 'medium' | 'low'
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    if len(df) < 7:
        # Not enough data for forecasting
        return pd.DataFrame(), {
            'next_value': 0,
            'trend_direction': 'unknown',
            'confidence': 'low'
        }
    
    # Prepare data for regression
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate residuals for confidence intervals
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_error = np.std(residuals)
    
    # Generate future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_ahead,
        freq='D'
    )
    
    # Predict future values
    future_X = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Calculate confidence intervals (±1.96 * std_error for 95% confidence)
    confidence_interval = 1.96 * std_error
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates,
        f'{metric}_forecast': future_y,
        f'{metric}_forecast_lower': future_y - confidence_interval,
        f'{metric}_forecast_upper': future_y + confidence_interval
    })
    
    # Calculate R² for confidence assessment
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Determine confidence level
    if r_squared > 0.7:
        confidence = 'high'
    elif r_squared > 0.4:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    forecast_stats = {
        'next_value': float(future_y[0]),
        'trend_direction': 'increasing' if model.coef_[0] > 0 else 'decreasing',
        'confidence': confidence,
        'r_squared': float(r_squared),
        'forecast_days': days_ahead
    }
    
    return forecast_df, forecast_stats

