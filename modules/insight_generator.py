"""
Insight generation module
Generates plain-language insights from analysis results
Includes ML-based pattern discovery
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


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
        # Count critical anomalies (severity >= 3.0) from the actual data
        from .anomaly_detector import get_critical_anomalies
        critical_anomalies = get_critical_anomalies(df, severity_threshold=3.0)
        critical_count = len(critical_anomalies)
        
        insights.append({
            'text': f"🔍 **{anomaly_summary['count']} anomalies detected** in your data "
                   f"({critical_count} critical). "
                   f"Most common types: {', '.join(list(anomaly_summary['by_type'].keys())[:3])}. "
                   f"See anomaly details below.",
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
    
    # ML-based pattern discovery insights
    ml_insights = discover_ml_patterns(df)
    insights.extend(ml_insights)
    
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


def discover_ml_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Discover patterns using ML algorithms (clustering, correlation, feature importance).
    
    This implements true automated insight generation using unsupervised ML
    to discover patterns without predefined rules.
    
    Args:
        df: DataFrame with metrics
    
    Returns:
        List of insight dictionaries from ML pattern discovery
    """
    insights = []
    
    try:
        # Aggregate by campaign for clustering
        from .data_processor import aggregate_by_campaign
        campaign_agg = aggregate_by_campaign(df)
        
        if len(campaign_agg) < 3:
            # Need at least 3 campaigns for meaningful clustering
            return insights
        
        # 1. K-Means Clustering: Discover campaign performance groups
        cluster_insights = discover_campaign_clusters(campaign_agg)
        insights.extend(cluster_insights)
        
        # 2. Correlation Analysis: Find unexpected relationships
        correlation_insights = discover_correlations(df)
        insights.extend(correlation_insights)
        
        # 3. Feature Importance: Identify which metrics matter most
        importance_insights = discover_feature_importance(df)
        insights.extend(importance_insights)
        
    except Exception as e:
        # Silently fail - ML insights are optional
        pass
    
    return insights


def determine_optimal_clusters_elbow(X_scaled: np.ndarray, max_clusters: int = 5) -> int:
    """
    Determine optimal number of clusters using the elbow method.
    
    The elbow method finds the optimal number of clusters by identifying
    the "elbow" point where adding more clusters doesn't significantly
    reduce within-cluster sum of squares (WCSS).
    
    Args:
        X_scaled: Normalized feature matrix
        max_clusters: Maximum number of clusters to test
    
    Returns:
        Optimal number of clusters (2 to max_clusters)
    """
    if len(X_scaled) < 2:
        return 1
    
    # Limit max_clusters to dataset size
    max_clusters = min(max_clusters, len(X_scaled) - 1)
    if max_clusters < 2:
        return 2
    
    # Calculate WCSS for different numbers of clusters
    wcss = []
    cluster_range = range(1, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)  # inertia_ = WCSS
    
    # Calculate rate of change (second derivative approximation)
    # Find the elbow: where the rate of decrease slows significantly
    if len(wcss) < 3:
        # Not enough points for elbow detection, use heuristic
        return min(2, max_clusters)
    
    # Calculate percentage decrease in WCSS
    decreases = []
    for i in range(1, len(wcss)):
        if wcss[i-1] > 0:
            pct_decrease = (wcss[i-1] - wcss[i]) / wcss[i-1] * 100
            decreases.append(pct_decrease)
        else:
            decreases.append(0)
    
    # Find elbow: largest drop in decrease rate
    # (i.e., where adding clusters stops helping much)
    if len(decreases) < 2:
        return 2
    
    # Calculate second-order differences (change in decrease rate)
    second_order = []
    for i in range(1, len(decreases)):
        second_order.append(decreases[i-1] - decreases[i])
    
    # Find elbow: where second-order difference is largest positive value
    # (i.e., where the benefit of adding clusters drops most)
    if len(second_order) == 0:
        return 2
    
    # If all second_order values are similar or negative, use minimum
    if all(d <= 0 for d in second_order) or max(second_order) < 5:
        # No clear elbow, use heuristic: 2-3 clusters for small datasets
        return min(3, max_clusters)
    
    # Elbow is at the point before the largest second-order change
    elbow_idx = second_order.index(max(second_order))
    optimal_k = elbow_idx + 2  # +2 because: k=1 at index 0, k=2 at index 1, etc.
    
    # Ensure reasonable range
    return max(2, min(optimal_k, max_clusters))


def discover_campaign_clusters(campaign_agg: pd.DataFrame) -> List[Dict]:
    """
    Use K-means clustering to discover groups of similar campaigns.
    
    Args:
        campaign_agg: Aggregated campaign data
    
    Returns:
        List of clustering insights
    """
    insights = []
    
    if len(campaign_agg) < 3:
        return insights
    
    # Features for clustering: ROI, CTR, conversion_rate, spend
    features = ['roi', 'ctr', 'conversion_rate', 'spend']
    available_features = [f for f in features if f in campaign_agg.columns]
    
    if len(available_features) < 2:
        return insights
    
    # Prepare data
    X = campaign_agg[available_features].fillna(0).values
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using elbow method
    n_clusters = determine_optimal_clusters_elbow(X_scaled, max_clusters=min(5, len(campaign_agg)))
    if n_clusters < 2:
        # Fallback: ensure at least 2 clusters
        n_clusters = min(2, len(campaign_agg))
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    campaign_agg['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    for cluster_id in range(n_clusters):
        cluster_campaigns = campaign_agg[campaign_agg['cluster'] == cluster_id]
        
        if len(cluster_campaigns) > 0:
            avg_roi = cluster_campaigns['roi'].mean()
            campaign_names = ', '.join(cluster_campaigns['campaign'].head(3).tolist())
            
            if len(cluster_campaigns) > 1:
                insights.append({
                    'text': f"🔍 **ML Pattern Discovery**: Campaigns {campaign_names} "
                           f"form a performance cluster with average ROI of {avg_roi:.2f}x. "
                           f"These campaigns show similar performance patterns and may benefit from similar optimization strategies.",
                    'priority': 3,
                    'type': 'info',
                    'category': 'ml_pattern'
                })
    
    return insights


def discover_correlations(df: pd.DataFrame) -> List[Dict]:
    """
    Discover unexpected correlations between metrics using statistical analysis.
    
    Args:
        df: DataFrame with metrics
    
    Returns:
        List of correlation insights
    """
    insights = []
    
    # Numeric columns for correlation
    numeric_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue', 
                   'roi', 'cpa', 'ctr', 'conversion_rate']
    available_cols = [c for c in numeric_cols if c in df.columns]
    
    if len(available_cols) < 3:
        return insights
    
    # Calculate correlation matrix
    corr_matrix = df[available_cols].corr()
    
    # Find interesting correlations (strong but not obvious)
    # Skip obvious ones (revenue vs conversions, etc.)
    interesting_pairs = []
    for i, col1 in enumerate(available_cols):
        for col2 in available_cols[i+1:]:
            # Skip obvious correlations
            if (col1 == 'revenue' and col2 == 'conversions') or \
               (col1 == 'conversions' and col2 == 'revenue') or \
               (col1 == 'spend' and col2 == 'revenue') or \
               (col1 == 'revenue' and col2 == 'spend'):
                continue
            
            corr_value = corr_matrix.loc[col1, col2]
            
            # Strong positive or negative correlation
            if abs(corr_value) > 0.6 and not np.isnan(corr_value):
                interesting_pairs.append({
                    'metric1': col1,
                    'metric2': col2,
                    'correlation': corr_value
                })
    
    # Sort by absolute correlation
    interesting_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Generate insights for top correlations
    for pair in interesting_pairs[:2]:  # Top 2
        direction = "positively" if pair['correlation'] > 0 else "negatively"
        strength = "strong" if abs(pair['correlation']) > 0.7 else "moderate"
        
        insights.append({
            'text': f"🔗 **ML-Discovered Correlation**: {pair['metric1'].replace('_', ' ').title()} and "
                   f"{pair['metric2'].replace('_', ' ').title()} show a {strength} {direction} correlation "
                   f"({pair['correlation']:.2f}). This suggests they may be related - consider analyzing them together.",
            'priority': 3,
            'type': 'info',
            'category': 'ml_pattern'
        })
    
    return insights


def discover_feature_importance(df: pd.DataFrame) -> List[Dict]:
    """
    Use Random Forest to discover which metrics are most important for predicting revenue.
    
    Args:
        df: DataFrame with metrics
    
    Returns:
        List of feature importance insights
    """
    insights = []
    
    # Features to analyze
    feature_cols = ['spend', 'impressions', 'clicks', 'conversions', 'ctr', 'conversion_rate']
    available_features = [f for f in feature_cols if f in df.columns]
    
    if 'revenue' not in df.columns or len(available_features) < 2:
        return insights
    
    # Prepare data
    X = df[available_features].fillna(0).values
    y = df['revenue'].fillna(0).values
    
    if len(X) < 10 or np.std(y) == 0:
        return insights
    
    # Train Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        # Get feature importance
        importances = rf.feature_importances_
        feature_importance = list(zip(available_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Top 2 most important features
        if len(feature_importance) >= 2:
            top_feature = feature_importance[0]
            second_feature = feature_importance[1]
            
            # Only report if importance is significant
            if top_feature[1] > 0.15:  # At least 15% importance
                insights.append({
                    'text': f"📊 **ML Feature Importance**: {top_feature[0].replace('_', ' ').title()} is the most "
                           f"important predictor of revenue ({top_feature[1]:.0%} importance), followed by "
                           f"{second_feature[0].replace('_', ' ').title()} ({second_feature[1]:.0%}). "
                           f"Focus optimization efforts on these metrics for maximum impact.",
                    'priority': 2,
                    'type': 'info',
                    'category': 'ml_pattern'
                })
    except Exception:
        # Silently fail
        pass
    
    return insights

