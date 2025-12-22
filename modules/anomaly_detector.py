"""
Anomaly detection module
Identifies statistical anomalies using Z-scores and ML-based pattern detection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


def detect_point_anomalies(
    df: pd.DataFrame,
    columns: List[str] = ['spend', 'conversions', 'revenue'],
    threshold: float = 2.5
) -> pd.DataFrame:
    """
    Detect point anomalies using Z-score method.
    
    Z-score formula: z = (x - μ) / σ
    Flag as anomaly if |z| > threshold
    
    Args:
        df: DataFrame with metrics
        columns: Columns to check for anomalies
        threshold: Z-score threshold (default 2.5 = ~1% outliers)
    
    Returns:
        DataFrame with added anomaly columns
    """
    df = df.copy()
    
    # Initialize anomaly flags
    df['is_anomaly'] = False
    df['anomaly_type'] = ''
    df['anomaly_severity'] = 0.0
    df['anomaly_details'] = ''
    
    for col in columns:
        if col in df.columns:
            # Calculate Z-scores
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:  # Avoid division by zero
                df[f'{col}_zscore'] = (df[col] - mean) / std
                
                # Flag anomalies
                high_anomalies = df[f'{col}_zscore'] > threshold
                low_anomalies = df[f'{col}_zscore'] < -threshold
                
                # Update anomaly flags
                df.loc[high_anomalies, 'is_anomaly'] = True
                df.loc[low_anomalies, 'is_anomaly'] = True
                
                # Track severity (max absolute Z-score)
                df.loc[high_anomalies | low_anomalies, 'anomaly_severity'] = df.loc[
                    high_anomalies | low_anomalies, 
                    'anomaly_severity'
                ].combine(
                    df.loc[high_anomalies | low_anomalies, f'{col}_zscore'].abs(),
                    max
                )
                
                # Build anomaly description
                df.loc[high_anomalies, 'anomaly_type'] = df.loc[high_anomalies, 'anomaly_type'] + f'High {col}; '
                df.loc[low_anomalies, 'anomaly_type'] = df.loc[low_anomalies, 'anomaly_type'] + f'Low {col}; '
                
                # Add details (build string for each row to avoid array formatting issues)
                for idx in df[high_anomalies].index:
                    val = df.loc[idx, col]
                    z_val = df.loc[idx, f'{col}_zscore']
                    df.loc[idx, 'anomaly_details'] += f'{col}: {val:.2f} (Z={z_val:.2f}); '
                
                for idx in df[low_anomalies].index:
                    val = df.loc[idx, col]
                    z_val = df.loc[idx, f'{col}_zscore']
                    df.loc[idx, 'anomaly_details'] += f'{col}: {val:.2f} (Z={z_val:.2f}); '
    
    # Also flag zero values as anomalies for conversions and revenue (always anomalous in marketing)
    # This ensures embedded anomalies with zero values are detected
    # But skip if already flagged by budget_exhaustion to avoid double-counting
    for col in ['conversions', 'revenue']:
        if col in df.columns:
            zero_mask = (df[col] == 0) & (~df.get('budget_exhaustion', pd.Series([False] * len(df), index=df.index)))
            df.loc[zero_mask, 'is_anomaly'] = True
            df.loc[zero_mask, 'anomaly_type'] = df.loc[zero_mask, 'anomaly_type'] + f'Zero {col}; '
            df.loc[zero_mask, 'anomaly_severity'] = df.loc[zero_mask, 'anomaly_severity'].combine(
                pd.Series([3.0] * zero_mask.sum(), index=df[zero_mask].index),
                max
            )
            df.loc[zero_mask, 'anomaly_details'] = df.loc[zero_mask, 'anomaly_details'] + f'{col}: 0.00 (Zero value); '
    
    # Clean up anomaly type (remove trailing semicolon)
    df['anomaly_type'] = df['anomaly_type'].str.rstrip('; ')
    df['anomaly_details'] = df['anomaly_details'].str.rstrip('; ')
    
    return df


def detect_gradual_anomalies(
    df: pd.DataFrame,
    metric: str = 'ctr',
    window: int = 7,
    decline_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Detect gradual performance degradation over time.
    
    Uses rolling window comparison to identify sustained declines.
    
    Args:
        df: DataFrame sorted by date
        metric: Column to analyze (e.g., 'ctr', 'conversion_rate')
        window: Days for rolling comparison
        decline_threshold: Minimum decline rate to flag (e.g., 0.08 = 8%)
    
    Returns:
        DataFrame with gradual_anomaly flags and decline percentages
    """
    df = df.copy()
    
    # Sort by date and campaign to analyze each campaign separately
    df = df.sort_values(['campaign', 'date'])
    
    # Initialize flags
    df['gradual_anomaly'] = False
    df['decline_pct'] = 0.0
    
    # Analyze each campaign separately
    for campaign in df['campaign'].unique():
        mask = df['campaign'] == campaign
        campaign_data = df[mask].copy()
        
        if len(campaign_data) >= window * 2:  # Need enough data
            # Calculate rolling average for first and last windows
            campaign_data = campaign_data.sort_values('date')
            first_window_avg = campaign_data[metric].iloc[:window].mean()
            last_window_avg = campaign_data[metric].iloc[-window:].mean()
            
            if first_window_avg > 0:
                decline = (first_window_avg - last_window_avg) / first_window_avg
                
                if decline >= decline_threshold:
                    # Flag the last window as gradual anomaly
                    last_window_indices = campaign_data.index[-window:]
                    df.loc[last_window_indices, 'gradual_anomaly'] = True
                    df.loc[last_window_indices, 'decline_pct'] = decline * 100
                    
                    # Add to anomaly tracking
                    df.loc[last_window_indices, 'is_anomaly'] = True
                    df.loc[last_window_indices, 'anomaly_type'] = df.loc[last_window_indices, 'anomaly_type'] + \
                        f'Gradual {metric} decline; '
                    
                    # Assign severity based on decline percentage (ensures it appears in critical list if >= 30% decline)
                    # Map decline % to severity: 8% = 2.5, 30% = 3.5, 40% = 4.0
                    decline_severity = 2.5 + min((decline / 0.08) * 1.0, 2.0)  # Scale from 2.5 to 4.5
                    df.loc[last_window_indices, 'anomaly_severity'] = df.loc[last_window_indices, 'anomaly_severity'].combine(
                        pd.Series([decline_severity] * len(last_window_indices), index=last_window_indices),
                        max
                    )
    
    return df


def detect_zero_spend_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect budget exhaustion (unexpected zero-spend periods).
    
    Flags days with zero spend when campaign should be active
    (determined by having non-zero spend in neighboring days).
    
    Args:
        df: DataFrame with spend column
    
    Returns:
        DataFrame with budget_exhaustion flags
    """
    df = df.copy()
    
    # Initialize flag
    df['budget_exhaustion'] = False
    
    # Sort by campaign and date
    df = df.sort_values(['campaign', 'date'])
    
    # Analyze each campaign
    for campaign in df['campaign'].unique():
        mask = df['campaign'] == campaign
        campaign_data = df[mask].copy().sort_values('date')
        
        # Find zero-spend days
        zero_spend = campaign_data['spend'] == 0
        
        if zero_spend.any():
            zero_spend_indices = campaign_data[zero_spend].index
            
            # Check if this is unexpected (has non-zero spend before and/or after)
            for idx in zero_spend_indices:
                idx_pos = campaign_data.index.get_loc(idx)
                
                # Check if campaign was active before or after
                has_activity_before = idx_pos > 0 and campaign_data.iloc[:idx_pos]['spend'].sum() > 0
                has_activity_after = idx_pos < len(campaign_data) - 1 and campaign_data.iloc[idx_pos+1:]['spend'].sum() > 0
                
                if has_activity_before or has_activity_after:
                    df.loc[idx, 'budget_exhaustion'] = True
                    df.loc[idx, 'is_anomaly'] = True
                    df.loc[idx, 'anomaly_type'] = df.loc[idx, 'anomaly_type'] + 'Budget exhaustion; '
    
    return df


def get_anomaly_summary(df: pd.DataFrame) -> Dict:
    """
    Summarize detected anomalies across all types.
    
    Args:
        df: DataFrame with anomaly flags
    
    Returns:
        Dictionary with anomaly statistics and examples
    """
    anomalies = df[df['is_anomaly'] == True].copy()
    
    if len(anomalies) == 0:
        return {
            'count': 0,
            'most_recent': [],
            'by_type': {},
            'by_category': {
                'point': 0,
                'gradual': 0,
                'budget': 0
            }
        }
    
    # Count by type
    by_type = {}
    for anomaly_types in anomalies['anomaly_type'].str.split('; '):
        for atype in anomaly_types:
            if atype:
                by_type[atype] = by_type.get(atype, 0) + 1
    
    # Categorize
    point_count = sum(1 for _ in anomalies[anomalies['anomaly_type'].str.contains('High |Low ', na=False)].iterrows())
    gradual_count = sum(1 for _ in anomalies[anomalies['gradual_anomaly'] == True].iterrows())
    budget_count = sum(1 for _ in anomalies[anomalies['budget_exhaustion'] == True].iterrows())
    
    # Get most recent anomalies (top 10)
    most_recent = anomalies.nlargest(10, 'date')[
        ['date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity', 'spend', 'conversions', 'revenue']
    ].to_dict('records')
    
    return {
        'count': len(anomalies),
        'most_recent': most_recent,
        'by_type': by_type,
        'by_category': {
            'point': point_count,
            'gradual': gradual_count,
            'budget': budget_count
        }
    }


def get_critical_anomalies(df: pd.DataFrame, severity_threshold: float = 3.0) -> pd.DataFrame:
    """
    Filter to only critical anomalies (high severity or specific types).
    
    Args:
        df: DataFrame with anomaly detection results
        severity_threshold: Minimum severity (Z-score) to be considered critical
    
    Returns:
        DataFrame with only critical anomalies
    """
    critical = df[
        (df['is_anomaly'] == True) &
        ((df['anomaly_severity'] >= severity_threshold) | (df['budget_exhaustion'] == True))
    ].copy()
    
    return critical.sort_values('date', ascending=False)


def detect_ml_anomalies(
    df: pd.DataFrame,
    columns: List[str] = ['spend', 'conversions', 'revenue', 'ctr'],
    contamination: float = 0.05
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest (ML-based algorithm).
    
    Isolation Forest is an unsupervised ML algorithm that identifies anomalies
    by isolating outliers in the feature space. This complements Z-score detection
    by finding complex multi-dimensional anomalies.
    
    Args:
        df: DataFrame with metrics
        columns: Columns to use for anomaly detection
        contamination: Expected proportion of anomalies (0.05 = 5%)
    
    Returns:
        DataFrame with added ML anomaly flags
    """
    df = df.copy()
    
    # Initialize ML anomaly flag if not exists
    if 'ml_anomaly' not in df.columns:
        df['ml_anomaly'] = False
        df['ml_anomaly_score'] = 0.0
    
    # Select available columns
    feature_cols = [col for col in columns if col in df.columns]
    
    if len(feature_cols) < 2:
        # Need at least 2 features for ML detection
        return df
    
    # Prepare features (handle NaN values)
    X = df[feature_cols].fillna(0).values
    
    if len(X) < 10:
        # Need minimum data points
        return df
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    predictions = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    # Update DataFrame
    df['ml_anomaly'] = predictions == -1
    df['ml_anomaly_score'] = -anomaly_scores  # Lower score = more anomalous
    
    # Merge with existing anomaly flags
    df.loc[df['ml_anomaly'], 'is_anomaly'] = True
    df.loc[df['ml_anomaly'], 'anomaly_type'] = df.loc[df['ml_anomaly'], 'anomaly_type'] + 'ML-detected; '
    
    # Update severity for ML anomalies (normalize score to 0-5 range)
    ml_mask = df['ml_anomaly']
    if ml_mask.any():
        min_score = df.loc[ml_mask, 'ml_anomaly_score'].min()
        max_score = df.loc[ml_mask, 'ml_anomaly_score'].max()
        if max_score > min_score:
            normalized_severity = 1 + 4 * (df.loc[ml_mask, 'ml_anomaly_score'] - min_score) / (max_score - min_score)
            df.loc[ml_mask, 'anomaly_severity'] = df.loc[ml_mask, 'anomaly_severity'].combine(
                normalized_severity, max
            )
    
    return df

