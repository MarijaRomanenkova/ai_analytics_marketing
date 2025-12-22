"""
ML-based budget optimization module
Uses ML predictions to recommend optimal budget allocation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def predict_channel_roi(
    df: pd.DataFrame,
    channel: str,
    days_ahead: int = 7
) -> Tuple[float, float]:
    """
    Predict future ROI for a channel using ML-based forecasting.
    
    Uses linear regression to predict revenue and spend trends,
    then calculates expected ROI with confidence bounds.
    
    Args:
        df: DataFrame with channel data
        channel: Channel name to predict
        days_ahead: Days to forecast ahead
    
    Returns:
        Tuple of (predicted_roi, confidence_score)
    """
    channel_df = df[df['channel'] == channel].copy()
    
    if len(channel_df) < 7:
        # Not enough data, use historical average
        if channel_df['spend'].sum() > 0:
            historical_roi = channel_df['revenue'].sum() / channel_df['spend'].sum()
            return historical_roi, 0.3  # Low confidence
    
    # Aggregate by date for trend analysis
    channel_by_date = channel_df.groupby('date').agg({
        'spend': 'sum',
        'revenue': 'sum'
    }).reset_index()
    channel_by_date = channel_by_date.sort_values('date')
    
    # Calculate daily ROI
    channel_by_date['daily_roi'] = np.where(
        channel_by_date['spend'] > 0,
        channel_by_date['revenue'] / channel_by_date['spend'],
        0
    )
    
    # Predict ROI trend using linear regression (ML)
    X = np.arange(len(channel_by_date)).reshape(-1, 1)
    y_roi = channel_by_date['daily_roi'].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y_roi)
    
    # Predict future ROI
    future_X = np.array([[len(channel_by_date) + days_ahead // 2]])
    predicted_roi = model.predict(future_X)[0]
    
    # Calculate confidence (R²)
    y_pred = model.predict(X)
    ss_res = np.sum((y_roi - y_pred) ** 2)
    ss_tot = np.sum((y_roi - np.mean(y_roi)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Ensure ROI is reasonable (not negative, not too high)
    predicted_roi = max(0.1, min(predicted_roi, 10.0))
    
    return float(predicted_roi), float(r_squared)


def optimize_budget_allocation(
    df: pd.DataFrame,
    total_budget: float = None,
    days_ahead: int = 7
) -> Dict:
    """
    Optimize budget allocation across channels using ML predictions.
    
    Uses ML-based ROI predictions to find the budget allocation that
    maximizes expected return. This is AI-powered optimization.
    
    Args:
        df: DataFrame with channel data
        total_budget: Total budget to allocate (if None, uses current total)
        days_ahead: Days ahead for ROI prediction
    
    Returns:
        Dictionary with optimization results:
        {
            'recommended_allocation': {channel: budget},
            'current_allocation': {channel: budget},
            'expected_roi_improvement': float,
            'expected_revenue_increase': float,
            'confidence': float
        }
    """
    # Get current channel performance
    channel_agg = df.groupby('channel').agg({
        'spend': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Calculate current total budget
    current_total = channel_agg['spend'].sum()
    if total_budget is None:
        total_budget = current_total
    
    if total_budget <= 0:
        return {
            'recommended_allocation': {},
            'current_allocation': {},
            'expected_roi_improvement': 0,
            'expected_revenue_increase': 0,
            'confidence': 0
        }
    
    # Predict ROI for each channel using ML
    channels = channel_agg['channel'].tolist()
    predicted_rois = {}
    confidences = {}
    
    for channel in channels:
        roi, confidence = predict_channel_roi(df, channel, days_ahead)
        predicted_rois[channel] = roi
        confidences[channel] = confidence
    
    # Current allocation
    current_allocation = dict(zip(
        channel_agg['channel'],
        channel_agg['spend']
    ))
    
    # Optimization function: maximize expected revenue
    # Expected revenue = sum(budget_i * predicted_roi_i)
    def objective(x):
        # Negative because we're minimizing
        expected_revenue = -sum(x[i] * predicted_rois[channels[i]] for i in range(len(channels)))
        return expected_revenue
    
    # Constraints: sum of allocations = total_budget
    constraints = {
        'type': 'eq',
        'fun': lambda x: sum(x) - total_budget
    }
    
    # Bounds: each channel gets at least 5% of budget, max 60%
    bounds = [
        (total_budget * 0.05, total_budget * 0.60)
        for _ in channels
    ]
    
    # Initial guess: proportional to predicted ROI
    initial_guess = []
    total_predicted_value = sum(predicted_rois.values())
    for channel in channels:
        if total_predicted_value > 0:
            initial_guess.append(total_budget * predicted_rois[channel] / total_predicted_value)
        else:
            initial_guess.append(total_budget / len(channels))
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_allocation = {
                channels[i]: max(0, result.x[i])
                for i in range(len(channels))
            }
        else:
            # Fallback: allocate proportionally to predicted ROI
            optimized_allocation = {}
            total_roi = sum(predicted_rois.values())
            for channel in channels:
                if total_roi > 0:
                    optimized_allocation[channel] = total_budget * predicted_rois[channel] / total_roi
                else:
                    optimized_allocation[channel] = total_budget / len(channels)
    except Exception:
        # Fallback to proportional allocation
        optimized_allocation = {}
        total_roi = sum(predicted_rois.values())
        for channel in channels:
            if total_roi > 0:
                optimized_allocation[channel] = total_budget * predicted_rois[channel] / total_roi
            else:
                optimized_allocation[channel] = total_budget / len(channels)
    
    # Calculate expected improvements
    current_expected_revenue = sum(
        current_allocation.get(ch, 0) * predicted_rois[ch]
        for ch in channels
    )
    
    optimized_expected_revenue = sum(
        optimized_allocation.get(ch, 0) * predicted_rois[ch]
        for ch in channels
    )
    
    current_roi = current_expected_revenue / current_total if current_total > 0 else 0
    optimized_roi = optimized_expected_revenue / total_budget if total_budget > 0 else 0
    
    # Average confidence across channels
    avg_confidence = np.mean(list(confidences.values())) if confidences else 0
    
    return {
        'recommended_allocation': optimized_allocation,
        'current_allocation': current_allocation,
        'predicted_rois': predicted_rois,
        'expected_roi_improvement': optimized_roi - current_roi,
        'expected_revenue_increase': optimized_expected_revenue - current_expected_revenue,
        'current_expected_roi': current_roi,
        'optimized_expected_roi': optimized_roi,
        'confidence': avg_confidence,
        'total_budget': total_budget
    }


def generate_optimization_insights(optimization_result: Dict) -> List[Dict]:
    """
    Generate natural language insights from optimization results.
    
    Args:
        optimization_result: From optimize_budget_allocation()
    
    Returns:
        List of insight dictionaries
    """
    insights = []
    
    if not optimization_result or optimization_result.get('expected_roi_improvement', 0) <= 0:
        return insights
    
    recommended = optimization_result['recommended_allocation']
    current = optimization_result['current_allocation']
    predicted_rois = optimization_result.get('predicted_rois', {})
    
    # Find biggest changes
    changes = []
    for channel in recommended.keys():
        current_budget = current.get(channel, 0)
        recommended_budget = recommended.get(channel, 0)
        change_pct = ((recommended_budget - current_budget) / current_budget * 100) if current_budget > 0 else 0
        
        if abs(change_pct) > 5:  # Only show significant changes
            changes.append({
                'channel': channel,
                'current': current_budget,
                'recommended': recommended_budget,
                'change_pct': change_pct,
                'predicted_roi': predicted_rois.get(channel, 0)
            })
    
    # Sort by absolute change
    changes.sort(key=lambda x: abs(x['change_pct']), reverse=True)
    
    # Generate insights
    if optimization_result['expected_roi_improvement'] > 0.1:
        insights.append({
            'text': f"💰 **ML-Optimized Budget Reallocation**: Reallocating budget based on AI predictions "
                   f"could improve ROI by **{optimization_result['expected_roi_improvement']:.2f}x** "
                   f"and increase expected revenue by **€{optimization_result['expected_revenue_increase']:,.0f}**.",
            'priority': 1,
            'type': 'success',
            'category': 'optimization'
        })
    
    # Top recommendations
    for change in changes[:3]:  # Top 3 changes
        direction = "increase" if change['change_pct'] > 0 else "decrease"
        insights.append({
            'text': f"📊 **{change['channel']}**: {direction.capitalize()} budget by "
                   f"**{abs(change['change_pct']):.0f}%** "
                   f"(€{change['current']:,.0f} → €{change['recommended']:,.0f}). "
                   f"Predicted ROI: {change['predicted_roi']:.2f}x",
            'priority': 2,
            'type': 'info',
            'category': 'optimization'
        })
    
    return insights

