"""
Insight generation module
Generates plain-language insights from analysis results
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime


def generate_insights(
    df: pd.DataFrame,
    anomaly_summary: Dict,
    trend_stats: Dict,
    date_range: Tuple[datetime, datetime] = None
) -> List[Dict]:
    """
    Generate prioritized natural language insights.
    
    Args:
        df: Processed data with metrics
        anomaly_summary: From anomaly_detector.get_anomaly_summary()
        trend_stats: From trend_analyzer.analyze_trends()
        date_range: (start_date, end_date) for context
    
    Returns:
        List of insight dicts sorted by priority
    """
    insights = []
    
    # Ensure we have channel aggregates
    from .data_processor import aggregate_by_channel, aggregate_by_campaign
    
    channel_agg = aggregate_by_channel(df)
    campaign_agg = aggregate_by_campaign(df)
    
    # 1. Best performing channel (Priority 1)
    if len(channel_agg) > 0:
        best_channel = channel_agg.iloc[0]
        insights.append({
            'text': f"🏆 **{best_channel['channel']}** is your top performer with **{best_channel['roi']:.2f}x ROI** "
                   f"(€{best_channel['revenue']:,.0f} revenue from €{best_channel['spend']:,.0f} spend). "
                   f"{'Consider increasing budget allocation here.' if best_channel['roi'] > 2 else ''}",
            'priority': 1,
            'type': 'success',
            'category': 'performance'
        })
    
    # 2. Worst performing channel (Priority 2) - if ROI < 1
    if len(channel_agg) > 0:
        worst_channel = channel_agg.iloc[-1]
        if worst_channel['roi'] < 1.0:
            insights.append({
                'text': f"⚠️ **{worst_channel['channel']}** is underperforming with **{worst_channel['roi']:.2f}x ROI** "
                       f"(losing €{worst_channel['spend'] - worst_channel['revenue']:,.0f}). "
                       f"Consider pausing or optimizing this channel.",
                'priority': 2,
                'type': 'warning',
                'category': 'performance'
            })
    
    # 3. Anomaly alerts (Priority 2)
    if anomaly_summary['count'] > 0:
        critical_count = sum(1 for a in anomaly_summary['most_recent'] 
                           if a.get('anomaly_severity', 0) >= 3.0)
        
        insights.append({
            'text': f"🔍 **{anomaly_summary['count']} anomalies detected** in your data "
                   f"({critical_count} critical). "
                   f"Most common types: {', '.join(list(anomaly_summary['by_type'].keys())[:3])}. "
                   f"Review the anomaly table below for details.",
            'priority': 2,
            'type': 'warning' if critical_count > 0 else 'info',
            'category': 'anomaly'
        })
        
        # Add specific anomaly insights for critical ones
        for anomaly in anomaly_summary['most_recent'][:3]:  # Top 3
            if anomaly.get('anomaly_severity', 0) >= 3.0:
                date_str = pd.to_datetime(anomaly['date']).strftime('%b %d')
                insights.append({
                    'text': f"⚠️ **{anomaly['campaign']}** on {date_str}: {anomaly['anomaly_type']} "
                           f"(severity: {anomaly['anomaly_severity']:.1f}). "
                           f"Immediate review recommended.",
                    'priority': 2,
                    'type': 'warning',
                    'category': 'anomaly'
                })
    
    # 4. Budget concentration warning (Priority 3)
    if len(campaign_agg) >= 3:
        total_spend = campaign_agg['spend'].sum()
        top_3_spend = campaign_agg.head(3)['spend'].sum()
        concentration = (top_3_spend / total_spend * 100) if total_spend > 0 else 0
        
        if concentration > 70:
            insights.append({
                'text': f"💰 **{concentration:.0f}% of budget** concentrated in top 3 campaigns. "
                       f"Consider diversifying to reduce risk and test new opportunities.",
                'priority': 3,
                'type': 'info',
                'category': 'budget'
            })
    
    # 5. Trend summary (Priority 3)
    if 'conversions' in trend_stats:
        conv_trend = trend_stats['conversions']
        if conv_trend['strength'] in ['strong', 'moderate']:
            direction_emoji = '📈' if conv_trend['direction'] == 'increasing' else '📉'
            insights.append({
                'text': f"{direction_emoji} **Conversions trending {conv_trend['direction']}** "
                       f"with {conv_trend['strength']} confidence "
                       f"({conv_trend['percent_change']:+.1f}% change). "
                       f"{'This positive trend suggests current strategies are working well.' if conv_trend['direction'] == 'increasing' else 'Consider investigating factors contributing to the decline.'}",
                'priority': 3,
                'type': 'success' if conv_trend['direction'] == 'increasing' else 'warning',
                'category': 'trend'
            })
    
    # 6. Day-of-week patterns (Priority 3)
    from .trend_analyzer import detect_day_of_week_patterns
    dow_patterns = detect_day_of_week_patterns(df)
    
    if dow_patterns['insights']:
        for insight_text in dow_patterns['insights']:
            insights.append({
                'text': f"📅 **{insight_text}**. Adjust bidding strategies or budget allocation accordingly.",
                'priority': 3,
                'type': 'info',
                'category': 'temporal'
            })
    
    # 7. Seasonal effects (Priority 3)
    from .trend_analyzer import detect_seasonal_patterns
    seasonal_insights = detect_seasonal_patterns(df)
    
    for seasonal_text in seasonal_insights:
        insights.append({
            'text': f"🎯 **{seasonal_text}**. Plan campaigns around seasonal patterns.",
            'priority': 3,
            'type': 'info',
            'category': 'temporal'
        })
    
    # 8. Campaign lifecycle alerts (Priority 3)
    lifecycle_insights = check_campaign_lifecycle(df, datetime.now() if date_range is None else date_range[1])
    for lifecycle_text in lifecycle_insights:
        insights.append({
            'text': f"📌 {lifecycle_text}",
            'priority': 3,
            'type': 'info',
            'category': 'lifecycle'
        })
    
    # 9. Gradual degradation warnings (Priority 2)
    gradual_anomalies = df[df.get('gradual_anomaly', False) == True]
    if len(gradual_anomalies) > 0:
        for campaign in gradual_anomalies['campaign'].unique():
            campaign_data = gradual_anomalies[gradual_anomalies['campaign'] == campaign]
            decline_pct = campaign_data['decline_pct'].iloc[0]
            insights.append({
                'text': f"⚠️ **{campaign}**: Gradual performance decline detected "
                       f"({decline_pct:.1f}% decrease). This suggests ad fatigue - "
                       f"consider refreshing creative or targeting.",
                'priority': 2,
                'type': 'warning',
                'category': 'anomaly'
            })
    
    # 10. ROI comparison insight (Priority 3)
    if len(channel_agg) > 1:
        roi_spread = channel_agg['roi'].max() - channel_agg['roi'].min()
        if roi_spread > 1.0:
            insights.append({
                'text': f"📊 **Large ROI variance** across channels ({roi_spread:.1f}x difference). "
                       f"Opportunities exist to reallocate budget from lower to higher performing channels.",
                'priority': 3,
                'type': 'info',
                'category': 'performance'
            })
    
    # Sort by priority (lower number = higher priority) and return
    insights.sort(key=lambda x: (x['priority'], x['text']))
    
    return insights


def check_campaign_lifecycle(df: pd.DataFrame, current_date: datetime) -> List[str]:
    """
    Check for campaigns starting/ending soon.
    
    Args:
        df: DataFrame with campaign data
        current_date: Reference date for "now"
    
    Returns:
        List of campaign lifecycle insight strings
    """
    insights = []
    
    # Find campaigns that have ended
    for campaign in df['campaign'].unique():
        campaign_data = df[df['campaign'] == campaign]
        last_date = campaign_data['date'].max()
        first_date = campaign_data['date'].min()
        
        # Check if campaign has ended (no data in last 7 days of dataset)
        days_since_last = (df['date'].max() - last_date).days
        
        if days_since_last > 7:
            insights.append(
                f"**{campaign}** ended on {last_date.strftime('%b %d')} "
                f"(ran for {(last_date - first_date).days} days). Performance data is complete."
            )
        
        # Check if campaign is new (started recently)
        days_since_start = (df['date'].max() - first_date).days
        if days_since_start <= 7:
            insights.append(
                f"**{campaign}** is new (started {first_date.strftime('%b %d')}). "
                f"Allow more time for performance stabilization."
            )
    
    return insights


def format_insight_for_display(insight: Dict) -> str:
    """
    Format insight dictionary for streamlit display.
    
    Args:
        insight: Insight dictionary
    
    Returns:
        Formatted markdown string
    """
    return insight['text']


def get_insights_by_category(insights: List[Dict], category: str) -> List[Dict]:
    """
    Filter insights by category.
    
    Args:
        insights: List of insight dictionaries
        category: Category to filter by
    
    Returns:
        Filtered list of insights
    """
    return [i for i in insights if i['category'] == category]


def get_critical_insights(insights: List[Dict]) -> List[Dict]:
    """
    Get only high-priority insights (priority 1-2).
    
    Args:
        insights: List of insight dictionaries
    
    Returns:
        Filtered list of critical insights
    """
    return [i for i in insights if i['priority'] <= 2]

