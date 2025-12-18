"""
Visualization module
Creates interactive Plotly charts for data visualization
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


def create_timeseries_chart(
    df: pd.DataFrame,
    metric: str,
    show_anomalies: bool = True,
    show_trend: bool = True,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create interactive time-series chart.
    
    Args:
        df: Data with date, metric, optional anomaly flags and trend
        metric: Column name to plot
        show_anomalies: Highlight anomalies with red markers
        show_trend: Show trend line and moving average
        title: Chart title (auto-generated if None)
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Main line chart
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[metric],
        mode='lines+markers',
        name=metric.replace('_', ' ').title(),
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>' +
                     f'{metric.replace("_", " ").title()}: %{{y:,.2f}}<br>' +
                     '<extra></extra>'
    ))
    
    # Add moving average if available
    ma_col = f'{metric}_ma7'
    if show_trend and ma_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[ma_col],
            mode='lines',
            name='7-Day Moving Avg',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>%{x|%b %d}</b><br>' +
                         '7-Day Avg: %{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Add trend line if available
    trend_col = f'{metric}_trend'
    if show_trend and trend_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[trend_col],
            mode='lines',
            name='Trend',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %d}</b><br>' +
                         'Trend: %{y:,.2f}<br>' +
                         '<extra></extra>'
        ))
    
    # Highlight anomalies
    if show_anomalies and 'is_anomaly' in df.columns:
        anomalies = df[df['is_anomaly'] == True]
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['date'],
                y=anomalies[metric],
                mode='markers',
                name='Anomalies',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='darkred')
                ),
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>' +
                             f'{metric.replace("_", " ").title()}: %{{y:,.2f}}<br>' +
                             'ANOMALY DETECTED<br>' +
                             '<extra></extra>'
            ))
    
    # Update layout
    chart_title = title or f'{metric.replace("_", " ").title()} Over Time'
    fig.update_layout(
        title=chart_title,
        xaxis_title='Date',
        yaxis_title=metric.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig


def create_channel_comparison_chart(channel_metrics: pd.DataFrame) -> go.Figure:
    """
    Create bar chart comparing channels.
    
    Args:
        channel_metrics: DataFrame with channel and ROI columns
    
    Returns:
        Plotly Figure object with horizontal bar chart
    """
    # Sort by ROI descending
    channel_metrics = channel_metrics.sort_values('roi', ascending=True)
    
    # Create color scale (green for good ROI, red for poor)
    colors = ['#d62728' if roi < 1 else '#2ca02c' if roi > 2 else '#ff7f0e' 
              for roi in channel_metrics['roi']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=channel_metrics['channel'],
        x=channel_metrics['roi'],
        orientation='h',
        marker=dict(color=colors),
        text=channel_metrics['roi'].apply(lambda x: f'{x:.2f}x'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>' +
                     'ROI: %{x:.2f}x<br>' +
                     '<extra></extra>'
    ))
    
    # Add vertical line at ROI = 1 (break-even)
    fig.add_vline(
        x=1,
        line_dash='dash',
        line_color='gray',
        annotation_text='Break-even',
        annotation_position='top right'
    )
    
    fig.update_layout(
        title='Channel Performance Comparison (ROI)',
        xaxis_title='Return on Investment (ROI)',
        yaxis_title='Channel',
        template='plotly_white',
        height=max(300, len(channel_metrics) * 60),
        showlegend=False
    )
    
    return fig


def create_campaign_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted campaign performance table.
    
    Args:
        df: Campaign-level aggregated data
    
    Returns:
        Styled DataFrame suitable for st.dataframe()
    """
    # Select and order columns
    table_columns = ['campaign', 'channel', 'spend', 'revenue', 'conversions', 'roi', 'ctr', 'conversion_rate']
    table_df = df[table_columns].copy()
    
    # Round numeric columns
    table_df['spend'] = table_df['spend'].round(2)
    table_df['revenue'] = table_df['revenue'].round(2)
    table_df['roi'] = table_df['roi'].round(2)
    table_df['ctr'] = table_df['ctr'].round(2)
    table_df['conversion_rate'] = table_df['conversion_rate'].round(2)
    
    # Rename columns for display
    table_df.columns = ['Campaign', 'Channel', 'Spend (€)', 'Revenue (€)', 'Conversions', 
                        'ROI', 'CTR (%)', 'Conv. Rate (%)']
    
    return table_df


def create_metric_distribution_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    """
    Create histogram showing distribution of a metric.
    
    Args:
        df: DataFrame with metric
        metric: Column name to plot
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[metric],
        nbinsx=30,
        marker=dict(color='#1f77b4', line=dict(color='white', width=1)),
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    mean_val = df[metric].mean()
    fig.add_vline(
        x=mean_val,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Mean: {mean_val:.2f}',
        annotation_position='top right'
    )
    
    fig.update_layout(
        title=f'Distribution of {metric.replace("_", " ").title()}',
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='Frequency',
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


def create_channel_spend_pie_chart(channel_metrics: pd.DataFrame) -> go.Figure:
    """
    Create pie chart showing spend distribution across channels.
    
    Args:
        channel_metrics: DataFrame with channel and spend columns
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=channel_metrics['channel'],
        values=channel_metrics['spend'],
        hole=0.3,
        hovertemplate='<b>%{label}</b><br>' +
                     'Spend: €%{value:,.0f}<br>' +
                     'Share: %{percent}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title='Budget Allocation by Channel',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create scatter plot highlighting anomalies.
    
    Args:
        df: DataFrame with anomaly flags
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Normal data points
    normal_data = df[df.get('is_anomaly', False) == False]
    fig.add_trace(go.Scatter(
        x=normal_data['spend'],
        y=normal_data['revenue'],
        mode='markers',
        name='Normal',
        marker=dict(size=6, color='#1f77b4', opacity=0.6),
        hovertemplate='<b>%{text}</b><br>' +
                     'Spend: €%{x:,.0f}<br>' +
                     'Revenue: €%{y:,.0f}<br>' +
                     '<extra></extra>',
        text=normal_data['campaign']
    ))
    
    # Anomaly data points
    anomaly_data = df[df.get('is_anomaly', False) == True]
    if len(anomaly_data) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_data['spend'],
            y=anomaly_data['revenue'],
            mode='markers',
            name='Anomaly',
            marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
            hovertemplate='<b>%{text}</b><br>' +
                         'Spend: €%{x:,.0f}<br>' +
                         'Revenue: €%{y:,.0f}<br>' +
                         'ANOMALY<br>' +
                         '<extra></extra>',
            text=anomaly_data['campaign']
        ))
    
    # Add break-even line (ROI = 1)
    max_val = max(df['spend'].max(), df['revenue'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Break-even (ROI=1)',
        line=dict(color='gray', dash='dash'),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title='Spend vs Revenue (Anomalies Highlighted)',
        xaxis_title='Spend (€)',
        yaxis_title='Revenue (€)',
        template='plotly_white',
        height=450,
        showlegend=True
    )
    
    return fig

