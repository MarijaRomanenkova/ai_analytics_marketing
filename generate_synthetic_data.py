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
    if not is_campaign_active(date, campaign):
        return None
    
    spend = config['base_spend'] * np.random.normal(1.0, 0.20)
    spend = max(spend, 10)
    
    day_of_week = date.weekday()
    if day_of_week == 0:
        spend *= 1.05
    elif day_of_week == 4:
        spend *= 0.90
    elif day_of_week >= 5:
        spend *= 0.70
    
    if date.month == 1 and date.day <= 15 and channel in ['Facebook', 'Instagram']:
        spend *= 1.5
    if date.month == 2 and 10 <= date.day <= 14 and channel == 'Instagram':
        spend *= 1.2
    if date.month == 2 and 20 <= date.day <= 28:
        spend *= 0.85
    
    if channel == 'Email':
        sends = int(spend / config['cost_per_send'])
        impressions = sends
        open_rate = np.random.uniform(*config['open_rate_range'])
        opens = int(impressions * open_rate)
        ctr = np.random.uniform(*config['ctr_range'])
        clicks = int(opens * ctr)
    else:
        cpm = np.random.uniform(*config['cpm_range'])
        impressions = int((spend / cpm) * 1000)
        ctr = np.random.uniform(*config['ctr_range'])
        clicks = int(impressions * ctr)
    
    clicks = max(clicks, 0)
    conv_rate = np.random.uniform(*config['conv_rate_range'])
    conversions = int(clicks * conv_rate) if clicks > 0 else 0
    conversions = max(conversions, 0)
    
    if date.weekday() >= 5:
        conversions = int(conversions * 0.6)
    if date.month == 1 and date.day <= 15 and channel in ['Facebook', 'Instagram']:
        conversions = int(conversions * 1.2)
    if date.month == 2 and 10 <= date.day <= 14 and channel == 'Instagram':
        conversions = int(conversions * 1.3)
    
    aov = np.random.uniform(*config['aov_range'])
    revenue = conversions * aov
    
    return {'date': date, 'channel': channel, 'campaign': campaign,
            'spend': round(spend, 2), 'impressions': impressions,
            'clicks': clicks, 'conversions': conversions, 'revenue': round(revenue, 2)}

def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    mask1 = (df['date'] == '2025-02-15') & (df['campaign'] == 'Google_Search_Brand')
    df.loc[mask1, 'spend'] *= 3.5
    df.loc[mask1, 'impressions'] = (df.loc[mask1, 'impressions'] * 3.5).astype(int)
    df.loc[mask1, 'clicks'] = (df.loc[mask1, 'clicks'] * 3.5).astype(int)
    
    mask2 = ((df['date'] >= '2025-03-10') & (df['date'] <= '2025-03-15') & (df['campaign'] == 'FB_Retargeting'))
    df.loc[mask2, 'conversions'] = (df.loc[mask2, 'conversions'] * 0.3).astype(int)
    df.loc[mask2, 'clicks'] = (df.loc[mask2, 'clicks'] * 0.5).astype(int)
    df.loc[mask2, 'revenue'] *= 0.3
    
    mask3 = (df['date'] == '2025-02-23') & (df['campaign'] == 'Email_Newsletter')
    df.loc[mask3, 'revenue'] *= 4.0
    df.loc[mask3, 'conversions'] = (df.loc[mask3, 'conversions'] * 3.0).astype(int)
    
    mask4 = ((df['date'] >= '2025-02-01') & (df['date'] <= '2025-02-28') & (df['campaign'] == 'IG_Influencer'))
    if mask4.sum() > 0:
        feb_start = pd.to_datetime('2025-02-01')
        df.loc[mask4, 'days_elapsed'] = (df.loc[mask4, 'date'] - feb_start).dt.days
        decline_factor = 1.0 - (df.loc[mask4, 'days_elapsed'] / 28 * 0.08)
        df.loc[mask4, 'clicks'] = (df.loc[mask4, 'clicks'] * decline_factor).astype(int)
        df.loc[mask4, 'conversions'] = (df.loc[mask4, 'conversions'] * decline_factor).astype(int)
        df.loc[mask4, 'revenue'] *= decline_factor
        df = df.drop(columns=['days_elapsed'], errors='ignore')
    
    mask5 = (df['date'].isin(['2025-03-22', '2025-03-23'])) & (df['campaign'] == 'Google_Display')
    df.loc[mask5, ['spend', 'impressions', 'clicks', 'conversions', 'revenue']] = 0
    
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
    df = inject_anomalies(df)
    df['date'] = pd.to_datetime(df['date'])
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
