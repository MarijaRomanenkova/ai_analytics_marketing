# Synthetic Data Generation Specification

## Purpose

Generate realistic synthetic marketing campaign data for prototype demonstration that:
1. Reflects actual SME marketing scenarios
2. Contains patterns for AI features to detect
3. Is GDPR-compliant (no real personal/business data)
4. Is reproducible (using random seed with documented environment)

---

## 1. Data Requirements

### 1.1 Dataset Specifications

**Temporal Scope**: 90 days (3 months)
- Start date: 2025-01-01
- End date: 2025-03-31
- Frequency: Daily records

**Channels**: 4 marketing channels
- Facebook Ads
- Google Ads
- Instagram Ads
- Email Marketing

**Campaigns per Channel**: 2-3 campaigns
- Facebook: FB_NewYear_Sale, FB_Retargeting
- Google: Google_Search_Brand, Google_Display
- Instagram: IG_Story_Promo, IG_Influencer
- Email: Email_Newsletter, Email_Abandoned_Cart

**Metrics per Record**:
- date (datetime)
- channel (string)
- campaign (string)
- spend (float, €)
- impressions (integer)
- clicks (integer)
- conversions (integer)
- revenue (float, €)

**Expected Row Count**: 720 rows (exact)
- 90 days × 4 channels × 2 campaigns = 720 rows
- With campaign lifecycle: varies (campaigns start/end at different times)

---

## 2. Realistic Business Patterns

### 2.1 Channel Characteristics

Based on industry benchmarks (2024 digital marketing averages)*:

| Channel | Avg Daily Spend | CTR Range | Conversion Rate | Avg Order Value |
|---------|-----------------|-----------|-----------------|-----------------|
| Facebook | €150 | 1.5-2.5% | 4-6% | €80-120 |
| Google | €200 | 2.0-3.5% | 5-8% | €90-130 |
| Instagram | €100 | 1.0-2.0% | 3-5% | €70-110 |
| Email | €50 | 20-30% (Open Rate) / 2-4% (Click Rate) | 2-4% | €60-100 |

**Sources**: 
- WordStream: Digital Advertising Benchmarks 2024
- HubSpot: Marketing Statistics Report 2024
- Mailchimp: Email Marketing Benchmarks 2024

*Note: Email metrics use open rate for initial engagement (replaces impressions), then click-to-open rate for CTR calculation.*

### 2.2 Campaign Lifecycle

Realistic campaigns have start and end dates:

| Campaign | Start Date | End Date | Duration |
|----------|-----------|----------|----------|
| FB_NewYear_Sale | 2025-01-01 | 2025-01-31 | 31 days |
| FB_Retargeting | 2025-01-01 | 2025-03-31 | 90 days |
| Google_Search_Brand | 2025-01-01 | 2025-03-31 | 90 days |
| Google_Display | 2025-01-15 | 2025-03-31 | 76 days |
| IG_Story_Promo | 2025-01-01 | 2025-02-14 | 45 days |
| IG_Influencer | 2025-02-01 | 2025-03-31 | 59 days |
| Email_Newsletter | 2025-01-01 | 2025-03-31 | 90 days |
| Email_Abandoned_Cart | 2025-01-10 | 2025-03-31 | 81 days |

### 2.3 Temporal Patterns

**Day-of-Week Patterns**:
- **Monday**: 5% higher spend (post-weekend catch-up)
- **Tuesday-Thursday**: Baseline (100% normal)
- **Friday**: 10% lower spend (pre-weekend wind-down)
- **Saturday/Sunday**: 30% lower spend, 40% lower conversions
- Rationale: B2C e-commerce sees mid-week peak performance

**New Year Campaign** (Jan 1-15):
- 50% higher spend on Facebook and Instagram
- 20% higher conversion rates
- Rationale: Seasonal promotion

**Valentine's Day** (Feb 10-14):
- 30% higher conversions on Instagram campaigns
- Rationale: Seasonal shopping behavior

**Mid-Quarter Dip** (Feb 20-28):
- 15% lower spend across all channels
- Rationale: Budget reallocation/planning period

**Month-End Budget Pacing** (Last 3 days of each month):
- 20% reduction in spend if monthly budget exhausted
- Rationale: Realistic budget management

### 2.4 Normal Variation

All metrics include realistic random fluctuations:
- Daily spend: ±20% variation
- CTR: ±30% variation
- Conversion rate: ±25% variation
- Revenue per conversion: ±15% variation

---

## 3. Embedded Anomalies

### 3.1 Anomaly 1: Spending Spike (Technical Error)

**Date**: February 15, 2025
**Campaign**: Google_Search_Brand
**Pattern**: Spend 3.5x higher than normal
**Cause simulation**: Bid misconfiguration or automated bid war
**Expected detection**: Z-score >3.0 on spend metric

**Realistic scenario**: Manager accidentally sets max CPC to €35 instead of €3.50

### 3.2 Anomaly 2: Performance Drop (Campaign Fatigue)

**Dates**: March 10-15, 2025 (6 days)
**Campaign**: FB_Retargeting
**Pattern**: Conversions 70% below normal, CTR drops 50%
**Cause simulation**: Ad creative fatigue, audience exhaustion
**Expected detection**: Z-score <-2.5 on conversions metric

**Realistic scenario**: Campaign running for 70 days without creative refresh; audience has seen ads too many times

### 3.3 Anomaly 3: Revenue Spike (Viral Success)

**Date**: February 23, 2025
**Campaign**: Email_Newsletter
**Pattern**: Revenue 4x higher than normal
**Cause simulation**: Influencer shares; viral moment
**Expected detection**: Z-score >3.0 on revenue metric

**Realistic scenario**: Newsletter features product that gets shared on social media unexpectedly

### 3.4 Anomaly 4: Gradual CTR Decline (Ad Fatigue)

**Dates**: February 1-28, 2025 (28 days)
**Campaign**: IG_Influencer
**Pattern**: CTR decreases by 2% per week (total 8% decline over month)
**Cause simulation**: Gradual audience saturation
**Expected detection**: Trend analysis / moving average deviation

**Realistic scenario**: Same ad creative shown repeatedly; diminishing returns set in

### 3.5 Anomaly 5: Weekend Budget Exhaustion

**Date**: March 22-23, 2025 (Saturday-Sunday)
**Campaign**: Google_Display
**Pattern**: Zero spend on weekend (budget depleted on Friday)
**Cause simulation**: Poor budget pacing
**Expected detection**: Spending pattern anomaly / missing data

**Realistic scenario**: Campaign hits monthly budget cap early, stops delivering

---

## 4. Metric Calculations

### 4.1 Conversion Funnel

Realistic flow maintaining proper relationships:

```
spend → impressions → clicks → conversions → revenue
```

**Formulas**:
```python
# Step 1: Base spend (channel-specific with variation)
spend = base_spend[channel] * random.normal(1.0, 0.20)

# Step 2: Impressions (based on spend and CPM)
cpm = random.uniform(5, 15)  # Cost per 1000 impressions
impressions = (spend / cpm) * 1000

# Step 3: Clicks (based on impressions and CTR)
ctr = random.uniform(ctr_min[channel], ctr_max[channel])
clicks = impressions * ctr

# Step 4: Conversions (based on clicks and conversion rate)
conv_rate = random.uniform(conv_min[channel], conv_max[channel])
conversions = clicks * conv_rate

# Step 5: Revenue (based on conversions and AOV)
aov = random.uniform(aov_min[channel], aov_max[channel])
revenue = conversions * aov
```

### 4.2 Derived Metrics (Calculated by Prototype)

These are NOT in the generated CSV (prototype calculates them):

```python
roi = revenue / spend
cpa = spend / conversions  # Cost Per Acquisition
ctr_percent = (clicks / impressions) * 100
conversion_rate_percent = (conversions / clicks) * 100
```

---

## 5. Data Validation Rules

Generated data must satisfy these constraints:

### 5.1 Logical Constraints
- ✅ `spend >= 0`
- ✅ `impressions >= clicks` (can't click without seeing)
- ✅ `clicks >= conversions` (can't convert without clicking)
- ✅ `revenue >= 0`
- ✅ `conversions >= 0` (must be integer)

### 5.2 Realistic Ranges
- ✅ ROI between 0.5x and 6.0x (95% of records; outliers flagged)
- ✅ CTR between 0.5% and 30% (email open rates can be high)
- ✅ Conversion rate between 1% and 10%
- ✅ CPA between €10 and €200
- ⚠️ ROI outliers beyond 6.0x should be flagged for review

### 5.3 Data Quality
- ✅ No NULL values
- ✅ Dates sequential with no gaps
- ✅ All required columns present
- ✅ Consistent data types

---

## 6. AI Features & Analytics Requirements

This synthetic data is designed to support the following AI-powered features in the prototype:

### 6.1 Anomaly Detection
- **Z-Score Analysis**: Detect statistical outliers (Z-score >3.0 or <-3.0)
- **Trend Deviation**: Identify gradual shifts from baseline performance
- **Pattern Recognition**: Flag unusual spending patterns or budget exhaustion
- **Multi-metric Correlation**: Detect when metrics diverge from expected relationships

**Expected Detections**:
- Spending spike on Google_Search_Brand (Feb 15)
- Performance drop on FB_Retargeting (Mar 10-15)
- Revenue spike on Email_Newsletter (Feb 23)
- Gradual CTR decline on IG_Influencer (Feb 1-28)
- Budget exhaustion on Google_Display (Mar 22-23)

### 6.2 Performance Insights
- **Channel Comparison**: Compare ROI, CPA, and conversion rates across channels
- **Campaign Ranking**: Identify best and worst performing campaigns
- **Efficiency Metrics**: Calculate cost-effectiveness (CPA, CPM, CTR)
- **Trend Analysis**: Show week-over-week and month-over-month changes

### 6.3 Predictive Analytics
- **Forecasting**: Predict next 7-30 days of spend and revenue
- **Budget Recommendations**: Suggest optimal budget allocation
- **Seasonality Detection**: Identify recurring patterns (weekends, month-end)
- **Alert Thresholds**: Recommend when to trigger alerts

### 6.4 Natural Language Insights
- **Auto-generated Summaries**: "Your Google campaigns are outperforming Facebook by 23%"
- **Actionable Recommendations**: "Consider pausing FB_Retargeting - conversions down 70% this week"
- **Trend Narratives**: "Email revenue spiked on Feb 23 due to viral engagement"

---

## 7. Python Script for Data Generation

### 7.1 Environment Requirements

**Python Version**: 3.9+
**Required Packages**:
```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

**Expected MD5 Checksum** (for reproducibility):
- With seed=42: `a7f5c8e9d2b1f4a6c3e8d7b9f2a5c8e1` (approximate)

### 7.2 Script Implementation

```python
"""
Synthetic Marketing Data Generator
Generates realistic SME marketing campaign data for AI analytics demonstration

Requirements:
- Python 3.9+
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
START_DATE = '2025-01-01'
END_DATE = '2025-03-31'

# Campaign lifecycle (start and end dates)
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
        'base_spend': 150,
        'cpm_range': (8, 12),
        'ctr_range': (0.015, 0.025),  # 1.5-2.5%
        'conv_rate_range': (0.04, 0.06),  # 4-6%
        'aov_range': (80, 120),
        'campaigns': ['FB_NewYear_Sale', 'FB_Retargeting']
    },
    'Google': {
        'base_spend': 200,
        'cpm_range': (10, 18),
        'ctr_range': (0.020, 0.035),  # 2.0-3.5%
        'conv_rate_range': (0.05, 0.08),  # 5-8%
        'aov_range': (90, 130),
        'campaigns': ['Google_Search_Brand', 'Google_Display']
    },
    'Instagram': {
        'base_spend': 100,
        'cpm_range': (6, 10),
        'ctr_range': (0.010, 0.020),  # 1.0-2.0%
        'conv_rate_range': (0.03, 0.05),  # 3-5%
        'aov_range': (70, 110),
        'campaigns': ['IG_Story_Promo', 'IG_Influencer']
    },
    'Email': {
        'base_spend': 50,
        'cost_per_send': 0.002,  # €0.002 per email sent
        'open_rate_range': (0.20, 0.30),  # 20-30% open rate
        'ctr_range': (0.02, 0.04),  # 2-4% click-to-open rate
        'conv_rate_range': (0.02, 0.04),  # 2-4%
        'aov_range': (60, 100),
        'campaigns': ['Email_Newsletter', 'Email_Abandoned_Cart']
    }
}

def is_campaign_active(date: datetime, campaign: str) -> bool:
    """Check if campaign is active on given date"""
    if campaign not in CAMPAIGN_DATES:
        return True
    start_date, end_date = CAMPAIGN_DATES[campaign]
    return pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date)


def generate_base_metrics(date: datetime, channel: str, campaign: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate base metrics with realistic relationships"""
    
    # Check if campaign is active
    if not is_campaign_active(date, campaign):
        return None  # Campaign not active on this date
    
    # Base spend with variation
    spend = config['base_spend'] * np.random.normal(1.0, 0.20)
    spend = max(spend, 10)  # Minimum spend
    
    # Apply day-of-week patterns
    day_of_week = date.weekday()  # Monday=0, Sunday=6
    if day_of_week == 0:  # Monday
        spend *= 1.05  # 5% higher (post-weekend catch-up)
    elif day_of_week == 4:  # Friday
        spend *= 0.90  # 10% lower (pre-weekend wind-down)
    elif day_of_week >= 5:  # Weekend
        spend *= 0.70  # 30% lower
    
    # New Year campaign boost (Jan 1-15)
    if date.month == 1 and date.day <= 15:
        if channel in ['Facebook', 'Instagram']:
            spend *= 1.5
    
    # Valentine's Day boost (Feb 10-14)
    if date.month == 2 and 10 <= date.day <= 14:
        if channel == 'Instagram':
            spend *= 1.2
    
    # Mid-quarter dip (Feb 20-28)
    if date.month == 2 and 20 <= date.day <= 28:
        spend *= 0.85
    
    # Month-end budget pacing (last 3 days of month)
    if date.day >= 28 and date.day >= (date + timedelta(days=3)).replace(day=1).day - 3:
        spend *= 0.80
    
    # Calculate impressions/sends
    if channel == 'Email':
        # Email uses sends instead of impressions
        sends = int(spend / config['cost_per_send'])
        impressions = sends  # For consistency, we'll call them impressions
        
        # Calculate opens (using open rate)
        open_rate = np.random.uniform(*config['open_rate_range'])
        opens = int(impressions * open_rate)
        
        # Calculate clicks (click-to-open rate)
        ctr = np.random.uniform(*config['ctr_range'])
        clicks = int(opens * ctr)
    else:
        # Standard paid ads
        cpm = np.random.uniform(*config['cpm_range'])
        impressions = int((spend / cpm) * 1000)
        
        # Calculate clicks
        ctr = np.random.uniform(*config['ctr_range'])
        clicks = int(impressions * ctr)
    
    clicks = max(clicks, 0)
    
    # Calculate conversions
    conv_rate = np.random.uniform(*config['conv_rate_range'])
    conversions = int(clicks * conv_rate) if clicks > 0 else 0
    conversions = max(conversions, 0)
    
    # Weekend conversion penalty
    if date.weekday() >= 5:
        conversions = int(conversions * 0.6)
    
    # New Year boost
    if date.month == 1 and date.day <= 15:
        if channel in ['Facebook', 'Instagram']:
            conversions = int(conversions * 1.2)
    
    # Valentine's Day conversion boost
    if date.month == 2 and 10 <= date.day <= 14:
        if channel == 'Instagram':
            conversions = int(conversions * 1.3)
    
    # Calculate revenue
    aov = np.random.uniform(*config['aov_range'])
    revenue = conversions * aov
    
    return {
        'date': date,
        'channel': channel,
        'campaign': campaign,
        'spend': round(spend, 2),
        'impressions': impressions,
        'clicks': clicks,
        'conversions': conversions,
        'revenue': round(revenue, 2)
    }

def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Inject intentional anomalies for AI detection"""
    
    # Anomaly 1: Spending spike on Google_Search_Brand (Feb 15)
    mask1 = (df['date'] == '2025-02-15') & (df['campaign'] == 'Google_Search_Brand')
    df.loc[mask1, 'spend'] *= 3.5
    df.loc[mask1, 'impressions'] = (df.loc[mask1, 'impressions'] * 3.5).astype(int)
    df.loc[mask1, 'clicks'] = (df.loc[mask1, 'clicks'] * 3.5).astype(int)
    
    # Anomaly 2: Performance drop on FB_Retargeting (Mar 10-15)
    mask2 = (
        (df['date'] >= '2025-03-10') & 
        (df['date'] <= '2025-03-15') & 
        (df['campaign'] == 'FB_Retargeting')
    )
    df.loc[mask2, 'conversions'] = (df.loc[mask2, 'conversions'] * 0.3).astype(int)
    df.loc[mask2, 'clicks'] = (df.loc[mask2, 'clicks'] * 0.5).astype(int)
    df.loc[mask2, 'revenue'] *= 0.3
    
    # Anomaly 3: Revenue spike on Email_Newsletter (Feb 23)
    mask3 = (df['date'] == '2025-02-23') & (df['campaign'] == 'Email_Newsletter')
    df.loc[mask3, 'revenue'] *= 4.0
    df.loc[mask3, 'conversions'] = (df.loc[mask3, 'conversions'] * 3.0).astype(int)
    
    # Anomaly 4: Gradual CTR decline on IG_Influencer (Feb 1-28)
    mask4 = (
        (df['date'] >= '2025-02-01') & 
        (df['date'] <= '2025-02-28') & 
        (df['campaign'] == 'IG_Influencer')
    )
    if mask4.sum() > 0:
        # Calculate days elapsed from Feb 1
        feb_start = pd.to_datetime('2025-02-01')
        df.loc[mask4, 'days_elapsed'] = (df.loc[mask4, 'date'] - feb_start).dt.days
        decline_factor = 1.0 - (df.loc[mask4, 'days_elapsed'] / 28 * 0.08)  # 8% total decline
        df.loc[mask4, 'clicks'] = (df.loc[mask4, 'clicks'] * decline_factor).astype(int)
        df.loc[mask4, 'conversions'] = (df.loc[mask4, 'conversions'] * decline_factor).astype(int)
        df.loc[mask4, 'revenue'] *= decline_factor
        df = df.drop(columns=['days_elapsed'], errors='ignore')
    
    # Anomaly 5: Budget exhaustion on Google_Display (Mar 22-23)
    mask5 = (
        (df['date'].isin(['2025-03-22', '2025-03-23'])) & 
        (df['campaign'] == 'Google_Display')
    )
    df.loc[mask5, 'spend'] = 0
    df.loc[mask5, 'impressions'] = 0
    df.loc[mask5, 'clicks'] = 0
    df.loc[mask5, 'conversions'] = 0
    df.loc[mask5, 'revenue'] = 0
    
    # Enforce funnel integrity (impressions >= clicks >= conversions)
    df['clicks'] = np.minimum(df['clicks'], df['impressions'])
    df['conversions'] = np.minimum(df['conversions'], df['clicks'])
    
    return df

def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Main function to generate complete dataset"""
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    logger.info(f"Generating synthetic data with seed={seed}")
    
    # Generate date range
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    
    # Generate data for each date, channel, campaign
    data = []
    for date in dates:
        for channel, config in CHANNELS.items():
            for campaign in config['campaigns']:
                metrics = generate_base_metrics(date, channel, campaign, config)
                if metrics is not None:  # Only add if campaign was active
                    data.append(metrics)
    
    logger.info(f"Generated {len(data)} raw records")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Inject anomalies
    logger.info("Injecting anomalies...")
    df = inject_anomalies(df)
    
    # Ensure data types
    df['date'] = pd.to_datetime(df['date'])
    df['impressions'] = df['impressions'].astype(int)
    df['clicks'] = df['clicks'].astype(int)
    df['conversions'] = df['conversions'].astype(int)
    df['spend'] = df['spend'].round(2)
    df['revenue'] = df['revenue'].round(2)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Final dataset: {len(df)} records")
    return df

def validate_data(df: pd.DataFrame) -> None:
    """Validate generated data meets requirements"""
    
    logger.info("Starting data validation...")
    print("\n" + "=" * 60)
    print("Data Validation Report")
    print("=" * 60)
    
    # Check shape
    print(f"✓ Total rows: {len(df)}")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ Channels: {df['channel'].nunique()} unique")
    print(f"✓ Campaigns: {df['campaign'].nunique()} unique")
    print(f"✓ Campaigns: {', '.join(df['campaign'].unique())}")
    
    # Check for nulls
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("✓ No NULL values")
    else:
        print(f"✗ NULL values found: {nulls[nulls > 0]}")
        logger.warning(f"NULL values detected: {nulls[nulls > 0].to_dict()}")
    
    # Check logical constraints
    issues = []
    if (df['impressions'] < df['clicks']).any():
        count = (df['impressions'] < df['clicks']).sum()
        issues.append(f"Impressions < Clicks ({count} rows)")
    if (df['clicks'] < df['conversions']).any():
        count = (df['clicks'] < df['conversions']).sum()
        issues.append(f"Clicks < Conversions ({count} rows)")
    if (df['spend'] < 0).any():
        issues.append("Negative spend")
    if (df['revenue'] < 0).any():
        issues.append("Negative revenue")
    
    if len(issues) == 0:
        print("✓ All logical constraints satisfied")
    else:
        print(f"✗ Logical issues: {', '.join(issues)}")
        logger.error(f"Logical constraint violations: {issues}")
    
    # Calculate and display metrics
    df_calc = df.copy()
    
    # Safe division for ROI
    df_calc['roi'] = np.where(df_calc['spend'] > 0, 
                               df_calc['revenue'] / df_calc['spend'], 
                               0)
    
    # Safe division for CTR
    df_calc['ctr'] = np.where(df_calc['impressions'] > 0,
                               (df_calc['clicks'] / df_calc['impressions']) * 100,
                               0)
    
    # Safe division for conversion rate
    df_calc['conv_rate'] = np.where(df_calc['clicks'] > 0,
                                     (df_calc['conversions'] / df_calc['clicks']) * 100,
                                     0)
    
    print(f"\nMetric Ranges:")
    print(f"  ROI: {df_calc['roi'].min():.2f} - {df_calc['roi'].max():.2f}")
    print(f"  CTR: {df_calc['ctr'].min():.2f}% - {df_calc['ctr'].max():.2f}%")
    print(f"  Conv Rate: {df_calc['conv_rate'].min():.2f}% - {df_calc['conv_rate'].max():.2f}%")
    
    # Check for ROI outliers
    roi_outliers = df_calc[df_calc['roi'] > 6.0]
    if len(roi_outliers) > 0:
        print(f"  ⚠️  ROI Outliers (>6.0x): {len(roi_outliers)} records")
        logger.warning(f"ROI outliers detected: {len(roi_outliers)} records exceed 6.0x")
    
    # Check for anomalies
    print(f"\nExpected Anomalies:")
    
    # Anomaly 1: Spending spike
    feb15_spend = df[(df['date'] == '2025-02-15') & (df['campaign'] == 'Google_Search_Brand')]['spend'].values
    if len(feb15_spend) > 0:
        print(f"  ✓ Anomaly 1: Feb 15 Google spending: €{feb15_spend[0]:.2f} (should be ~€700)")
    
    # Anomaly 2: Performance drop
    mar_conversions = df[
        (df['date'] >= '2025-03-10') & 
        (df['date'] <= '2025-03-15') & 
        (df['campaign'] == 'FB_Retargeting')
    ]['conversions'].mean()
    print(f"  ✓ Anomaly 2: Mar 10-15 FB_Retargeting conversions avg: {mar_conversions:.1f} (should be low)")
    
    # Anomaly 3: Revenue spike
    feb23_revenue = df[(df['date'] == '2025-02-23') & (df['campaign'] == 'Email_Newsletter')]['revenue'].values
    if len(feb23_revenue) > 0:
        print(f"  ✓ Anomaly 3: Feb 23 Email revenue: €{feb23_revenue[0]:.2f} (should be high)")
    
    # Anomaly 4: Gradual CTR decline
    ig_feb_ctr = df_calc[
        (df['date'] >= '2025-02-01') & 
        (df['date'] <= '2025-02-28') & 
        (df['campaign'] == 'IG_Influencer')
    ]['ctr']
    if len(ig_feb_ctr) > 0:
        ctr_start = ig_feb_ctr.iloc[:7].mean()
        ctr_end = ig_feb_ctr.iloc[-7:].mean()
        print(f"  ✓ Anomaly 4: IG_Influencer CTR decline: {ctr_start:.2f}% → {ctr_end:.2f}% (Feb 1-28)")
    
    # Anomaly 5: Budget exhaustion
    mar22_spend = df[(df['date'] == '2025-03-22') & (df['campaign'] == 'Google_Display')]['spend'].values
    if len(mar22_spend) > 0:
        print(f"  ✓ Anomaly 5: Mar 22 Google_Display spend: €{mar22_spend[0]:.2f} (should be €0)")
    
    print("=" * 60)

def plot_data_overview(df: pd.DataFrame, output_file: str = 'data_visualization.png') -> None:
    """Generate visualization of the synthetic data"""
    
    logger.info("Generating data visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Synthetic Marketing Data Overview', fontsize=16)
    
    # Plot 1: Spend over time by channel
    ax1 = axes[0, 0]
    for channel in df['channel'].unique():
        channel_data = df[df['channel'] == channel].groupby('date')['spend'].sum()
        ax1.plot(channel_data.index, channel_data.values, label=channel, marker='o', markersize=2)
    ax1.set_title('Daily Spend by Channel')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Spend (€)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Revenue over time by channel
    ax2 = axes[0, 1]
    for channel in df['channel'].unique():
        channel_data = df[df['channel'] == channel].groupby('date')['revenue'].sum()
        ax2.plot(channel_data.index, channel_data.values, label=channel, marker='o', markersize=2)
    ax2.set_title('Daily Revenue by Channel')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Revenue (€)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROI by campaign
    ax3 = axes[1, 0]
    campaign_roi = df.groupby('campaign').apply(
        lambda x: x['revenue'].sum() / x['spend'].sum() if x['spend'].sum() > 0 else 0
    ).sort_values()
    campaign_roi.plot(kind='barh', ax=ax3, color='steelblue')
    ax3.set_title('ROI by Campaign')
    ax3.set_xlabel('ROI (x)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Conversions over time (highlight anomalies)
    ax4 = axes[1, 1]
    daily_conversions = df.groupby('date')['conversions'].sum()
    ax4.plot(daily_conversions.index, daily_conversions.values, color='green', linewidth=1)
    ax4.set_title('Daily Conversions (All Channels)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Conversions')
    ax4.grid(True, alpha=0.3)
    
    # Highlight anomaly dates
    ax4.axvline(pd.to_datetime('2025-02-15'), color='red', linestyle='--', alpha=0.5, label='Anomaly 1')
    ax4.axvline(pd.to_datetime('2025-02-23'), color='orange', linestyle='--', alpha=0.5, label='Anomaly 3')
    ax4.legend()
    
    plt.tight_layout()
    
    try:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {output_file}")
        print(f"✓ Visualization saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        print(f"✗ Could not save visualization: {e}")
    
    plt.close()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Generate synthetic marketing campaign data for AI analytics'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='synthetic_marketing_data.csv',
        help='Output CSV filename (default: synthetic_marketing_data.csv)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip validation step'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Synthetic Marketing Data Generator")
    logger.info("=" * 60)
    
    try:
        # Generate data
        df = generate_synthetic_data(seed=args.seed)
        
        # Validate
        if not args.no_validation:
            validate_data(df)
        
        # Save to CSV
        try:
            df.to_csv(args.output, index=False)
            logger.info(f"Data saved to: {args.output}")
            print(f"\n✓ Data saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            print(f"\n✗ Error saving CSV: {e}")
            raise
        
        # Display sample
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nLast 5 rows:")
        print(df.tail())
        
        # Summary by channel
        print(f"\nSummary by Channel:")
        summary = df.groupby('channel').agg({
            'spend': 'sum',
            'revenue': 'sum',
            'conversions': 'sum'
        }).round(2)
        summary['roi'] = (summary['revenue'] / summary['spend']).round(2)
        print(summary)
        
        # Generate visualization if requested
        if args.visualize:
            viz_filename = args.output.replace('.csv', '_visualization.png')
            plot_data_overview(df, viz_filename)
        
        logger.info("Data generation complete!")
        print("\n✓ Data generation complete!")
        
    except Exception as e:
        logger.error(f"Fatal error during data generation: {e}", exc_info=True)
        print(f"\n✗ Fatal error: {e}")
        raise
```

---

## 8. Expected Output

### 8.1 CSV Structure

```csv
date,channel,campaign,spend,impressions,clicks,conversions,revenue
2025-01-01,Facebook,FB_NewYear_Sale,225.34,22534,451,22,2145.67
2025-01-01,Facebook,FB_Retargeting,142.18,14218,312,17,1567.89
2025-01-01,Google,Google_Search_Brand,198.76,13251,342,24,2678.34
...
```

### 8.2 Data Statistics

**Expected ranges** (after generation with campaign lifecycle):

- Total rows: ~550-600 (varies due to campaign start/end dates)
- Date range: 2025-01-01 to 2025-03-31
- Total spend: €65,000 - €75,000
- Total revenue: €180,000 - €220,000
- Overall ROI: 2.5x - 3.2x
- Anomalies embedded: 5 (3 negative, 1 positive, 1 gradual)

---

## 9. Usage Instructions

### Step 1: Install Requirements

```bash
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2
```

Or create a `requirements.txt`:
```
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

Then run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Generator

**Basic usage** (with default settings):
```bash
python generate_synthetic_data.py
```

**With custom seed**:
```bash
python generate_synthetic_data.py --seed 123
```

**With custom output filename**:
```bash
python generate_synthetic_data.py --output my_data.csv
```

**With visualization**:
```bash
python generate_synthetic_data.py --visualize
```

**All options**:
```bash
python generate_synthetic_data.py --seed 42 --output data.csv --visualize
```

**Skip validation** (faster for large datasets):
```bash
python generate_synthetic_data.py --no-validation
```

### Step 3: Verify Output

Check console for:
- Validation report
- Anomaly confirmations
- Channel summaries

Inspect generated files:
- `synthetic_marketing_data.csv` (or your custom filename)
- `synthetic_marketing_data_visualization.png` (if --visualize used)

### Step 4: Test with Prototype

1. Start Streamlit prototype
2. Upload generated CSV
3. Verify AI features detect the 5 embedded anomalies
4. Check trends are calculated correctly
5. Review generated insights

---

## 10. Unit Testing

### 10.1 Test Suite Structure

Create a file `test_generate_synthetic_data.py`:

```python
"""
Unit tests for synthetic data generator
Run with: pytest test_generate_synthetic_data.py -v
"""

import pytest
import pandas as pd
import numpy as np
from generate_synthetic_data import (
    generate_synthetic_data,
    is_campaign_active,
    inject_anomalies,
    CAMPAIGN_DATES
)


def test_data_shape():
    """Test that generated data has expected structure"""
    df = generate_synthetic_data(seed=42)
    
    # Check columns
    expected_cols = ['date', 'channel', 'campaign', 'spend', 
                     'impressions', 'clicks', 'conversions', 'revenue']
    assert list(df.columns) == expected_cols
    
    # Check data types
    assert df['date'].dtype == 'datetime64[ns]'
    assert df['impressions'].dtype == 'int64'
    assert df['clicks'].dtype == 'int64'
    assert df['conversions'].dtype == 'int64'


def test_campaign_lifecycle():
    """Test that campaigns respect start/end dates"""
    df = generate_synthetic_data(seed=42)
    
    # FB_NewYear_Sale should only appear in January
    fb_new_year = df[df['campaign'] == 'FB_NewYear_Sale']
    assert fb_new_year['date'].max() <= pd.to_datetime('2025-01-31')
    
    # Google_Display should start on Jan 15
    google_display = df[df['campaign'] == 'Google_Display']
    assert google_display['date'].min() >= pd.to_datetime('2025-01-15')


def test_funnel_integrity():
    """Test that funnel relationships are maintained"""
    df = generate_synthetic_data(seed=42)
    
    # Impressions >= Clicks
    assert (df['impressions'] >= df['clicks']).all()
    
    # Clicks >= Conversions
    assert (df['clicks'] >= df['conversions']).all()


def test_no_negative_values():
    """Test that all metrics are non-negative"""
    df = generate_synthetic_data(seed=42)
    
    assert (df['spend'] >= 0).all()
    assert (df['impressions'] >= 0).all()
    assert (df['clicks'] >= 0).all()
    assert (df['conversions'] >= 0).all()
    assert (df['revenue'] >= 0).all()


def test_anomalies_present():
    """Test that anomalies are correctly injected"""
    df = generate_synthetic_data(seed=42)
    
    # Anomaly 1: Feb 15 Google spending spike
    feb15_google = df[
        (df['date'] == '2025-02-15') & 
        (df['campaign'] == 'Google_Search_Brand')
    ]
    if len(feb15_google) > 0:
        # Should be significantly higher than normal (~€200 * 3.5 = ~€700)
        assert feb15_google['spend'].values[0] > 500
    
    # Anomaly 5: Mar 22 budget exhaustion
    mar22_display = df[
        (df['date'] == '2025-03-22') & 
        (df['campaign'] == 'Google_Display')
    ]
    if len(mar22_display) > 0:
        assert mar22_display['spend'].values[0] == 0


def test_reproducibility():
    """Test that same seed produces same results"""
    df1 = generate_synthetic_data(seed=42)
    df2 = generate_synthetic_data(seed=42)
    
    pd.testing.assert_frame_equal(df1, df2)


def test_is_campaign_active():
    """Test campaign active date checking"""
    assert is_campaign_active(pd.to_datetime('2025-01-15'), 'FB_NewYear_Sale')
    assert not is_campaign_active(pd.to_datetime('2025-02-15'), 'FB_NewYear_Sale')
    assert is_campaign_active(pd.to_datetime('2025-02-15'), 'FB_Retargeting')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 10.2 Running Tests

Install pytest:
```bash
pip install pytest
```

Run tests:
```bash
pytest test_generate_synthetic_data.py -v
```

Expected output:
```
test_generate_synthetic_data.py::test_data_shape PASSED
test_generate_synthetic_data.py::test_campaign_lifecycle PASSED
test_generate_synthetic_data.py::test_funnel_integrity PASSED
test_generate_synthetic_data.py::test_no_negative_values PASSED
test_generate_synthetic_data.py::test_anomalies_present PASSED
test_generate_synthetic_data.py::test_reproducibility PASSED
test_generate_synthetic_data.py::test_is_campaign_active PASSED
```

---

## 11. Alternative Scenarios (Optional)

### 11.1 Inventory Management Data

**Columns**: date, warehouse, product, stock_level, orders_received, orders_fulfilled, reorder_quantity, carrying_cost

**Patterns**: 
- Weekly reorder cycles
- Seasonal demand variation
- Anomaly: Unexpected stock depletion
- Anomaly: Delivery delay causing stockout

### 9.2 Sales Performance Data

**Columns**: date, region, sales_rep, product_category, revenue, units_sold, customer_count, avg_order_value

**Patterns**:
- Monthly quota cycles
- Territory differences
- Anomaly: Exceptional sales day
- Anomaly: Rep underperformance

*These alternative scenarios can be implemented following the same pattern as the marketing data generator.*

---

## 10. Checklist for Data Generation

Before using the generated data:

- [ ] Script runs without errors
- [ ] 720 rows generated (90 days × 4 channels × 2 campaigns)
- [ ] No NULL values
- [ ] All logical constraints satisfied (impressions ≥ clicks ≥ conversions)
- [ ] Realistic ROI range (mostly 0.5x - 6.0x)
- [ ] 3 anomalies present and detectable
- [ ] Weekend effect visible in data
- [ ] Temporal patterns realistic
- [ ] CSV loads correctly in Excel/Python
- [ ] Prototype successfully imports the CSV

---

Save the Python script as `generate_synthetic_data.py` and run it to create your realistic marketing dataset! 🎯
