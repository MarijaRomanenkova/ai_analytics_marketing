"""
Synthetic Marketing Data Generator
Generates realistic SME marketing campaign data for AI analytics demonstration
"""

import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
START_DATE = '2025-01-01'
END_DATE = '2025-03-31'

# Campaign lifecycle
CAMPAIGN_DATES = {
    'FB_NewYear_Sale': ('2025-01-01', '2025-01-31'),
    'FB_Retargeting': ('2025-01-01', '2025-03-31'),
    'Google_Search_Brand': ('2025-01-01', '2025-03-31'),
    'Google_Display': ('2025-01-15', '2025-03-31'),
    'IG_Story_Promo': ('2025-01-01', '2025-02-14'),
    'IG_Influencer': ('2025-02-01', '2025-03-31'),
    'Email_Newsletter': ('2025-01-01', '2025-03-31'),
    'Email_Abandoned_Cart': ('2025-01-10', '2025-03-31'),
}

# Channel configurations
CHANNELS = {
    'Facebook': {
        'base_spend': 150, 'cpm_range': (8, 12),
        'ctr_range': (0.015, 0.025), 'conv_rate_range': (0.04, 0.06),
        'aov_range': (80, 120), 'campaigns': ['FB_NewYear_Sale', 'FB_Retargeting']
    },
    'Google': {
        'base_spend': 200, 'cpm_range': (10, 18),
        'ctr_range': (0.020, 0.035), 'conv_rate_range': (0.05, 0.08),
        'aov_range': (90, 130), 'campaigns': ['Google_Search_Brand', 'Google_Display']
    },
    'Instagram': {
        'base_spend': 100, 'cpm_range': (6, 10),
        'ctr_range': (0.010, 0.020), 'conv_rate_range': (0.03, 0.05),
        'aov_range': (70, 110), 'campaigns': ['IG_Story_Promo', 'IG_Influencer']
    },
    'Email': {
        'base_spend': 50, 'cost_per_send': 0.002,
        'open_rate_range': (0.20, 0.30), 'ctr_range': (0.02, 0.04),
        'conv_rate_range': (0.02, 0.04), 'aov_range': (60, 100),
        'campaigns': ['Email_Newsletter', 'Email_Abandoned_Cart']
    }
}

def is_campaign_active(date: datetime, campaign: str) -> bool:
    if campaign not in CAMPAIGN_DATES:
        return True
    start_date, end_date = CAMPAIGN_DATES[campaign]
    return pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date)

def generate_base_metrics(date: datetime, channel: str, campaign: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate base metrics with FIXED values (deterministic) to ensure only
    the 5 embedded anomalies are detected, not random outliers.
    
    Uses mid-point values from ranges instead of random sampling.
    """
    if not is_campaign_active(date, campaign):
        return None
    
    # Use fixed base spend (no random variation)
    spend = float(config['base_spend'])
    
    # Apply day-of-week patterns (deterministic)
    day_of_week = date.weekday()
    if day_of_week == 0:
        spend *= 1.05
    elif day_of_week == 4:
        spend *= 0.90
    elif day_of_week >= 5:
        spend *= 0.70
    
    # Apply seasonal patterns (deterministic)
    if date.month == 1 and date.day <= 15 and channel in ['Facebook', 'Instagram']:
        spend *= 1.5
    if date.month == 2 and 10 <= date.day <= 14 and channel == 'Instagram':
        spend *= 1.2
    if date.month == 2 and 20 <= date.day <= 28:
        spend *= 0.85
    
    # Use fixed mid-point values instead of random
    if channel == 'Email':
        sends = int(spend / config['cost_per_send'])
        impressions = sends
        open_rate = np.mean(config['open_rate_range'])  # Mid-point instead of random
        opens = int(impressions * open_rate)
        ctr = np.mean(config['ctr_range'])  # Mid-point instead of random
        clicks = int(opens * ctr)
    else:
        cpm = np.mean(config['cpm_range'])  # Mid-point instead of random
        impressions = int((spend / cpm) * 1000)
        ctr = np.mean(config['ctr_range'])  # Mid-point instead of random
        clicks = int(impressions * ctr)
    
    clicks = max(clicks, 0)
    conv_rate = np.mean(config['conv_rate_range'])  # Mid-point instead of random
    conversions = int(clicks * conv_rate) if clicks > 0 else 0
    conversions = max(conversions, 0)
    
    # Apply day-of-week conversion patterns (deterministic)
    if date.weekday() >= 5:
        conversions = int(conversions * 0.6)
    if date.month == 1 and date.day <= 15 and channel in ['Facebook', 'Instagram']:
        conversions = int(conversions * 1.2)
    if date.month == 2 and 10 <= date.day <= 14 and channel == 'Instagram':
        conversions = int(conversions * 1.3)
    
    aov = np.mean(config['aov_range'])  # Mid-point instead of random
    revenue = conversions * aov
    
    return {'date': date, 'channel': channel, 'campaign': campaign,
            'spend': round(spend, 2), 'impressions': impressions,
            'clicks': clicks, 'conversions': conversions, 'revenue': round(revenue, 2)}

def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date column is datetime and convert to date string for reliable comparison
    df['date'] = pd.to_datetime(df['date'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')  # Add string column for comparison
    
    # Anomaly 1: Spending spike on Google_Search_Brand (Feb 15)
    mask1 = (df['date_str'] == '2025-02-15') & (df['campaign'] == 'Google_Search_Brand')
    if mask1.sum() > 0:
        spend_before = df.loc[mask1, 'spend'].values[0]
        df.loc[mask1, 'spend'] = df.loc[mask1, 'spend'] * 3.5
        spend_after = df.loc[mask1, 'spend'].values[0]
        df.loc[mask1, 'impressions'] = (df.loc[mask1, 'impressions'] * 3.5).astype(int)
        df.loc[mask1, 'clicks'] = (df.loc[mask1, 'clicks'] * 3.5).astype(int)
        logger.info(f"Anomaly 1 applied: {mask1.sum()} row(s) modified - Spend {spend_before:.2f} -> {spend_after:.2f} (x{spend_after/spend_before:.2f})")
    else:
        logger.warning(f"Anomaly 1: No rows matched! Looking for Feb 15, Google_Search_Brand")
    
    # Anomaly 2: Performance drop on FB_Retargeting (Mar 10-15)
    # Set to 0 to ensure detection (zero values are always outliers)
    mask2 = ((df['date_str'] >= '2025-03-10') & 
             (df['date_str'] <= '2025-03-15') & 
             (df['campaign'] == 'FB_Retargeting'))
    if mask2.sum() > 0:
        df.loc[mask2, 'conversions'] = 0  # Set to 0 for guaranteed detection
        df.loc[mask2, 'clicks'] = (df.loc[mask2, 'clicks'] * 0.1).astype(int)  # Also reduce clicks
        df.loc[mask2, 'revenue'] = 0.0  # Set revenue to 0
        logger.info(f"Anomaly 2 applied: {mask2.sum()} row(s) modified - Conversions/revenue set to 0")
    else:
        logger.warning(f"Anomaly 2: No rows matched! Looking for Mar 10-15, FB_Retargeting")
    
    # Anomaly 3: Revenue spike on Email_Newsletter (Feb 23)
    # Made extremely high (50x) to ensure detection even with global Z-score calculation
    mask3 = (df['date_str'] == '2025-02-23') & (df['campaign'] == 'Email_Newsletter')
    if mask3.sum() > 0:
        revenue_before = df.loc[mask3, 'revenue'].values[0]
        df.loc[mask3, 'revenue'] = df.loc[mask3, 'revenue'] * 50.0  # Extremely high: 50x to create global outlier
        revenue_after = df.loc[mask3, 'revenue'].values[0]
        df.loc[mask3, 'conversions'] = (df.loc[mask3, 'conversions'] * 40.0).astype(int)  # Match revenue increase
        logger.info(f"Anomaly 3 applied: {mask3.sum()} row(s) modified - Revenue {revenue_before:.2f} -> {revenue_after:.2f} (x{revenue_after/revenue_before:.2f})")
    else:
        logger.warning(f"Anomaly 3: No rows matched! Looking for Feb 23, Email_Newsletter")
    
    # Anomaly 4: Gradual CTR decline on IG_Influencer (entire campaign period)
    # Apply to entire campaign so first vs last window comparison detects it
    mask4 = df['campaign'] == 'IG_Influencer'
    if mask4.sum() > 0:
        campaign_start = df.loc[mask4, 'date'].min()
        df.loc[mask4, 'days_elapsed'] = (df.loc[mask4, 'date'] - campaign_start).dt.days
        max_days = df.loc[mask4, 'days_elapsed'].max()
        # Apply 40% decline over entire campaign period to ensure detection
        decline_factor = 1.0 - (df.loc[mask4, 'days_elapsed'] / max_days * 0.40) if max_days > 0 else 1.0
        df.loc[mask4, 'clicks'] = (df.loc[mask4, 'clicks'] * decline_factor).astype(int)
        df.loc[mask4, 'conversions'] = (df.loc[mask4, 'conversions'] * decline_factor).astype(int)
        df.loc[mask4, 'revenue'] = df.loc[mask4, 'revenue'] * decline_factor
        df = df.drop(columns=['days_elapsed'], errors='ignore')
        logger.info(f"Anomaly 4 applied: {mask4.sum()} row(s) modified - Gradual 40% decline over entire campaign")
    else:
        logger.warning(f"Anomaly 4: No rows matched! Looking for Feb 1-28, IG_Influencer")
    
    # Anomaly 5: Budget exhaustion on Google_Display (Mar 22-23)
    mask5 = (df['date_str'].isin(['2025-03-22', '2025-03-23'])) & (df['campaign'] == 'Google_Display')
    if mask5.sum() > 0:
        df.loc[mask5, ['spend', 'impressions', 'clicks', 'conversions', 'revenue']] = 0
        logger.info(f"Anomaly 5 applied: {mask5.sum()} row(s) modified - Budget exhaustion")
    else:
        logger.warning(f"Anomaly 5: No rows matched! Looking for Mar 22-23, Google_Display")
    
    # Remove temporary date_str column
    df = df.drop(columns=['date_str'], errors='ignore')
    
    df['clicks'] = np.minimum(df['clicks'], df['impressions'])
    df['conversions'] = np.minimum(df['conversions'], df['clicks'])
    
    return df

def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    logger.info(f"Generating synthetic data with seed={seed}")
    
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    data = []
    for date in dates:
        for channel, config in CHANNELS.items():
            for campaign in config['campaigns']:
                metrics = generate_base_metrics(date, channel, campaign, config)
                if metrics is not None:
                    data.append(metrics)
    
    df = pd.DataFrame(data)
    df = inject_anomalies(df)  # This now handles date conversion
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Generated {len(df)} records")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic marketing data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='synthetic_marketing_data.csv')
    args = parser.parse_args()
    
    df = generate_synthetic_data(seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"✓ Generated {len(df)} rows -> {args.output}")
    print(f"\nFirst 5 rows:\n{df.head()}")
