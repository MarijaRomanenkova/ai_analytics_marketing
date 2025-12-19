"""
SME AI-Assisted Analytics Prototype
Modules package initialization
"""

from .data_loader import load_csv, validate_columns
from .data_processor import (
    calculate_metrics, 
    filter_data, 
    aggregate_by_channel,
    aggregate_by_campaign,
    aggregate_by_date
)
from .anomaly_detector import (
    detect_point_anomalies,
    detect_gradual_anomalies,
    detect_zero_spend_anomalies,
    detect_ml_anomalies,
    get_anomaly_summary
)
from .trend_analyzer import analyze_trends, analyze_multiple_metrics, forecast_metric
from .insight_generator import generate_insights
from .visualizer import (
    create_timeseries_chart,
    create_channel_comparison_chart,
    create_campaign_table,
    create_metric_distribution_chart,
    create_channel_spend_pie_chart,
    create_anomaly_scatter
)

__all__ = [
    'load_csv',
    'validate_columns',
    'calculate_metrics',
    'filter_data',
    'aggregate_by_channel',
    'aggregate_by_campaign',
    'aggregate_by_date',
    'detect_point_anomalies',
    'detect_gradual_anomalies',
    'detect_zero_spend_anomalies',
    'detect_ml_anomalies',
    'get_anomaly_summary',
    'analyze_trends',
    'analyze_multiple_metrics',
    'forecast_metric',
    'generate_insights',
    'create_timeseries_chart',
    'create_channel_comparison_chart',
    'create_campaign_table',
    'create_metric_distribution_chart',
    'create_channel_spend_pie_chart',
    'create_anomaly_scatter'
]

