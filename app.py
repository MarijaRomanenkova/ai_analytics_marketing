"""
SME AI-Assisted Marketing Analytics Prototype
Main Streamlit Application

This prototype demonstrates custom AI analytics capabilities for SMEs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import our modules
from modules import (
    load_csv,
    calculate_metrics,
    filter_data,
    aggregate_by_channel,
    aggregate_by_campaign,
    aggregate_by_date,
    detect_point_anomalies,
    detect_gradual_anomalies,
    detect_zero_spend_anomalies,
    detect_ml_anomalies,
    get_anomaly_summary,
    analyze_trends,
    analyze_multiple_metrics,
    forecast_metric,
    generate_insights,
    optimize_budget_allocation,
    generate_optimization_insights,
    create_timeseries_chart,
    create_channel_comparison_chart,
    create_campaign_table,
    create_metric_distribution_chart,
    create_channel_spend_pie_chart,
    create_anomaly_scatter
)

# Page configuration
st.set_page_config(
    page_title="SME AI Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.df_processed = None


def is_embedded_anomaly(row):
    """Check if a row is part of one of the 5 embedded anomalies"""
    date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
    campaign = row['campaign']
    
    # Anomaly 1: Feb 15, Google_Search_Brand
    if date_str == '2025-02-15' and campaign == 'Google_Search_Brand':
        return '1: Spending spike'
    
    # Anomaly 2: Mar 10-15, FB_Retargeting
    if campaign == 'FB_Retargeting':
        anomaly_date = pd.to_datetime(date_str)
        if pd.to_datetime('2025-03-10') <= anomaly_date <= pd.to_datetime('2025-03-15'):
            return '2: Performance drop'
    
    # Anomaly 3: Feb 23, Email_Newsletter
    if date_str == '2025-02-23' and campaign == 'Email_Newsletter':
        return '3: Revenue spike'
    
    # Anomaly 4: IG_Influencer gradual decline (last 7 days of campaign: Mar 25-31)
    if campaign == 'IG_Influencer':
        anomaly_date = pd.to_datetime(date_str)
        if pd.to_datetime('2025-03-25') <= anomaly_date <= pd.to_datetime('2025-03-31'):
            return '4: Gradual decline'
    
    # Anomaly 5: Mar 22-23, Google_Display
    if campaign == 'Google_Display' and date_str in ['2025-03-22', '2025-03-23']:
        return '5: Budget exhaustion'
    
    return ''


def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🔍 AI-Assisted Marketing Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent insights for your marketing campaigns</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your marketing data (CSV)",
            type=['csv'],
            help="Required columns: date, channel, campaign, spend, impressions, clicks, conversions, revenue"
        )
        
        if uploaded_file is not None:
            try:
                # Load and validate data
                with st.spinner("Loading data..."):
                    df = load_csv(uploaded_file)
                    df_with_metrics = calculate_metrics(df)
                    st.session_state.df = df_with_metrics
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} rows successfully!")
                
            except ValueError as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.data_loaded = False
                return
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                st.session_state.data_loaded = False
                return
        
        # Show filters only if data is loaded
        if st.session_state.data_loaded:
            st.markdown("---")
            st.header("📊 Filters")
            
            df = st.session_state.df
            
            # Date range filter
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            
            date_range = st.date_input(
                "📅 Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Select the date range to analyze"
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = min_date, max_date
            
            # Channel filter
            all_channels = sorted(df['channel'].unique().tolist())
            selected_channels = st.multiselect(
                "📺 Channels",
                options=all_channels,
                default=all_channels,
                help="Select channels to include in analysis"
            )
            
            # Campaign filter (show only active campaigns in date range)
            from modules.data_processor import get_active_campaigns_for_date_range
            active_campaigns = get_active_campaigns_for_date_range(df, start_date, end_date)
            
            selected_campaigns = st.multiselect(
                "🎯 Campaigns",
                options=active_campaigns,
                default=active_campaigns,
                help="Select campaigns to include in analysis"
            )
            
            # Apply filters
            df_filtered = filter_data(
                df,
                start_date=start_date,
                end_date=end_date,
                channels=selected_channels if selected_channels else None,
                campaigns=selected_campaigns if selected_campaigns else None
            )
            
            st.session_state.df_processed = df_filtered
            
            # Show data summary
            st.markdown("---")
            st.caption(f"📊 {len(df_filtered)} rows selected")
    
    # Main content
    if not st.session_state.data_loaded:
        # Show welcome message
        st.info("👈 Please upload a CSV file to get started")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your CSV should contain these columns:
            - **date**: Date of the record (YYYY-MM-DD)
            - **channel**: Marketing channel (e.g., Facebook, Google, Instagram, Email)
            - **campaign**: Campaign name
            - **spend**: Amount spent (€)
            - **impressions**: Number of impressions
            - **clicks**: Number of clicks
            - **conversions**: Number of conversions
            - **revenue**: Revenue generated (€)
            
            💡 **Tip**: You can generate synthetic test data using the `generate_synthetic_data.py` script included in this project.
            """)
        
        return
    
    # Process data with AI features
    df = st.session_state.df_processed
    
    if len(df) == 0:
        st.warning("⚠️ No data matches your filter criteria. Please adjust your filters.")
        return
    
    # Run AI analysis
    with st.spinner("🤖 Running AI analysis..."):
        # Anomaly detection (statistical + ML)
        df = detect_point_anomalies(df)
        df = detect_gradual_anomalies(df)
        df = detect_zero_spend_anomalies(df)
        # ML detection with low contamination (1%) to primarily detect embedded anomalies
        df = detect_ml_anomalies(df, contamination=0.01)  # ML-based anomaly detection
        anomaly_summary = get_anomaly_summary(df)
        
        # Trend analysis
        df_by_date = aggregate_by_date(df)
        df_by_date, trend_stats = analyze_multiple_metrics(df_by_date, ['conversions', 'revenue', 'spend'])
        
        # ML-based forecasting
        forecast_df, forecast_stats = forecast_metric(df_by_date, metric='conversions', days_ahead=7)
        
        # ML-based budget optimization
        optimization_result = optimize_budget_allocation(df)
        
        # Generate insights
        insights = generate_insights(
            df,
            anomaly_summary,
            trend_stats.get('conversions', {}),
            date_range=(df['date'].min(), df['date'].max())
        )
        
        # Add optimization insights
        optimization_insights = generate_optimization_insights(optimization_result)
        insights.extend(optimization_insights)
        insights.sort(key=lambda x: x['priority'])
    
    # Display insights panel
    st.header("💡 Key Insights")
    
    if len(insights) > 0:
        # Show insights in columns
        col1, col2 = st.columns(2)
        
        for idx, insight in enumerate(insights[:6]):  # Show top 6
            col = col1 if idx % 2 == 0 else col2
            with col:
                if insight['type'] == 'success':
                    st.success(insight['text'])
                elif insight['type'] == 'warning':
                    st.warning(insight['text'])
                else:
                    st.info(insight['text'])
        
        # Show remaining insights in expander
        if len(insights) > 6:
            with st.expander(f"📊 View {len(insights) - 6} more insights"):
                for insight in insights[6:]:
                    if insight['type'] == 'success':
                        st.success(insight['text'])
                    elif insight['type'] == 'warning':
                        st.warning(insight['text'])
                    else:
                        st.info(insight['text'])
    else:
        st.info("No significant insights detected. This could mean your campaigns are performing consistently.")
    
    # Anomaly summary table (if anomalies detected)
    if anomaly_summary['count'] > 0:
        st.subheader("🔍 Detected Anomalies")
        
        from modules.anomaly_detector import get_critical_anomalies
        critical_anomalies = get_critical_anomalies(df, severity_threshold=3.0)
        all_anomalies = df[df['is_anomaly'] == True]
        
        if len(critical_anomalies) > 0:
            # Show critical anomalies table
            critical_df = critical_anomalies[
                ['date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity', 'spend', 'conversions', 'revenue']
            ].sort_values('anomaly_severity', ascending=False).head(10)  # Top 10 critical
            
            st.caption(f"Showing {len(critical_df)} most critical anomalies (severity ≥ 3.0) out of {len(all_anomalies)} total. See '🔍 Anomalies' tab for full details.")
            
            st.dataframe(
                critical_df.style.format({
                    'spend': '€{:,.2f}',
                    'revenue': '€{:,.2f}',
                    'conversions': '{:,.0f}',
                    'anomaly_severity': '{:.2f}'
                }),
                use_container_width=True,
                height=300
            )
        else:
            # If no critical, show summary of all
            st.caption(f"All {len(all_anomalies)} anomalies have low severity. See '🔍 Anomalies' tab for details.")
            
            summary_df = all_anomalies[
                ['date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity']
            ].sort_values('anomaly_severity', ascending=False).head(10)
            
            st.dataframe(
                summary_df.style.format({
                    'anomaly_severity': '{:.2f}'
                }),
                use_container_width=True,
                height=300
            )
    
    # Key metrics
    st.header("📈 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = df['spend'].sum()
    total_revenue = df['revenue'].sum()
    total_conversions = df['conversions'].sum()
    overall_roi = total_revenue / total_spend if total_spend > 0 else 0
    
    with col1:
        st.metric(
            "Total Spend",
            f"€{total_spend:,.0f}",
            help="Total amount spent across all selected campaigns"
        )
    
    with col2:
        st.metric(
            "Total Revenue",
            f"€{total_revenue:,.0f}",
            help="Total revenue generated"
        )
    
    with col3:
        st.metric(
            "Total Conversions",
            f"{total_conversions:,.0f}",
            help="Total number of conversions"
        )
    
    with col4:
        roi_delta = f"{(overall_roi - 1) * 100:+.0f}%" if overall_roi != 0 else None
        st.metric(
            "Overall ROI",
            f"{overall_roi:.2f}x",
            delta=roi_delta,
            help="Return on Investment (Revenue / Spend)"
        )
    
    # Visualizations
    st.header("📊 Performance Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Trends", "🏆 Channels", "🎯 Campaigns", "🔍 Anomalies", "🔮 Forecast", "💰 Optimization"])
    
    with tab1:
        st.subheader("Conversions Over Time")
        if 'conversions' in df_by_date.columns:
            fig_conversions = create_timeseries_chart(
                df_by_date,
                metric='conversions',
                show_anomalies=False,
                show_trend=True
            )
            st.plotly_chart(fig_conversions, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Over Time")
            if 'revenue' in df_by_date.columns:
                fig_revenue = create_timeseries_chart(
                    df_by_date,
                    metric='revenue',
                    show_anomalies=False,
                    show_trend=True
                )
                st.plotly_chart(fig_revenue, use_container_width=True)
        
        with col2:
            st.subheader("Spend Over Time")
            if 'spend' in df_by_date.columns:
                fig_spend = create_timeseries_chart(
                    df_by_date,
                    metric='spend',
                    show_anomalies=False,
                    show_trend=True
                )
                st.plotly_chart(fig_spend, use_container_width=True)
    
    with tab2:
        st.subheader("Channel Performance Comparison")
        
        channel_agg = aggregate_by_channel(df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_channel_roi = create_channel_comparison_chart(channel_agg)
            st.plotly_chart(fig_channel_roi, use_container_width=True)
        
        with col2:
            fig_spend_pie = create_channel_spend_pie_chart(channel_agg)
            st.plotly_chart(fig_spend_pie, use_container_width=True)
        
        # Channel metrics table
        st.subheader("Channel Metrics")
        st.dataframe(
            channel_agg[['channel', 'spend', 'revenue', 'conversions', 'roi', 'ctr', 'conversion_rate']].style.format({
                'spend': '€{:,.2f}',
                'revenue': '€{:,.2f}',
                'conversions': '{:,.0f}',
                'roi': '{:.2f}x',
                'ctr': '{:.2f}%',
                'conversion_rate': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Campaign Performance")
        
        campaign_agg = aggregate_by_campaign(df)
        
        # Campaign table
        campaign_table = create_campaign_table(campaign_agg)
        st.dataframe(
            campaign_table.style.format({
                'Spend (€)': '€{:,.2f}',
                'Revenue (€)': '€{:,.2f}',
                'Conversions': '{:,.0f}',
                'ROI': '{:.2f}x',
                'CTR (%)': '{:.2f}%',
                'Conv. Rate (%)': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = campaign_agg.to_csv(index=False)
        st.download_button(
            label="📥 Download Campaign Data",
            data=csv,
            file_name="campaign_performance.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("Anomaly Detection Results")
        
        if anomaly_summary['count'] > 0:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Anomalies", anomaly_summary['count'])
            with col2:
                st.metric("Point Anomalies", anomaly_summary['by_category']['point'])
            with col3:
                st.metric("Budget Issues", anomaly_summary['by_category']['budget'])
            
            # Anomaly scatter plot
            st.subheader("Spend vs Revenue (Anomalies Highlighted)")
            fig_anomaly = create_anomaly_scatter(df)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly details table with severity filter
            st.subheader("Anomaly Details")
            
            # Filter by severity
            from modules.anomaly_detector import get_critical_anomalies
            critical_anomalies = get_critical_anomalies(df, severity_threshold=3.0)
            all_anomalies = df[df['is_anomaly'] == True]
            
            filter_option = st.radio(
                "Filter anomalies:",
                ["Critical only (High severity)", "All anomalies"],
                index=0,
                horizontal=True
            )
            
            if filter_option == "Critical only (High severity)":
                anomalies_df = critical_anomalies[
                    ['date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity', 'spend', 'conversions', 'revenue']
                ].sort_values('anomaly_severity', ascending=False)
                st.info(f"Showing {len(anomalies_df)} critical anomalies (severity ≥ 3.0) out of {len(all_anomalies)} total. Focus on these for immediate action.")
            else:
                anomalies_df = all_anomalies[
                    ['date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity', 'spend', 'conversions', 'revenue']
                ].sort_values('anomaly_severity', ascending=False)
                st.info(f"Showing all {len(anomalies_df)} anomalies. Use 'Critical only' filter to focus on high-priority issues.")
            
            # Add embedded anomaly marker column
            anomalies_df['Embedded Anomaly'] = anomalies_df.apply(is_embedded_anomaly, axis=1)
            
            # Reorder columns to show embedded marker first
            cols = ['Embedded Anomaly', 'date', 'campaign', 'channel', 'anomaly_type', 'anomaly_severity', 'spend', 'conversions', 'revenue']
            anomalies_df = anomalies_df[cols]
            
            st.dataframe(
                anomalies_df.style.format({
                    'spend': '€{:,.2f}',
                    'revenue': '€{:,.2f}',
                    'conversions': '{:,.0f}',
                    'anomaly_severity': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ No anomalies detected! Your campaigns are performing consistently.")
    
    with tab5:
        st.subheader("🔮 AI-Powered Forecast")
        
        if len(forecast_df) > 0:
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Next Day Prediction",
                    f"{forecast_stats['next_value']:.0f} conversions",
                    help="Predicted conversions for tomorrow"
                )
            
            with col2:
                direction_emoji = "📈" if forecast_stats['trend_direction'] == 'increasing' else "📉"
                st.metric(
                    "Trend Direction",
                    f"{direction_emoji} {forecast_stats['trend_direction'].title()}",
                    help="Predicted trend direction"
                )
            
            with col3:
                confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
                st.metric(
                    "Confidence",
                    f"{confidence_colors.get(forecast_stats['confidence'], '⚪')} {forecast_stats['confidence'].title()}",
                    help=f"Model confidence (R² = {forecast_stats['r_squared']:.2f})"
                )
            
            # Forecast chart
            st.subheader("7-Day Forecast")
            
            # Get historical data with moving average and trend
            historical = df_by_date[['date', 'conversions']].copy()
            
            # Create visualization
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Historical data (actual values) - show as lighter/more transparent
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['conversions'],
                mode='lines+markers',
                name='Historical (Actual)',
                line=dict(color='#1f77b4', width=1.5),
                marker=dict(size=4, opacity=0.6),
                opacity=0.7
            ))
            
            # Add moving average on historical data if available
            if 'conversions_ma7' in df_by_date.columns:
                fig.add_trace(go.Scatter(
                    x=df_by_date['date'],
                    y=df_by_date['conversions_ma7'],
                    mode='lines',
                    name='7-Day Moving Avg (Historical)',
                    line=dict(color='#9467bd', width=2, dash='dot'),
                    opacity=0.8
                ))
            
            # Add regression trend line on historical data (extended to show projection)
            if 'conversions_trend' in df_by_date.columns:
                # Historical trend line
                fig.add_trace(go.Scatter(
                    x=df_by_date['date'],
                    y=df_by_date['conversions_trend'],
                    mode='lines',
                    name='Regression Trend (Historical)',
                    line=dict(color='#2ca02c', width=2, dash='dot'),
                    opacity=0.7
                ))
                
                # Extend trend line into forecast period for comparison
                last_date = df_by_date['date'].max()
                last_trend_value = df_by_date['conversions_trend'].iloc[-1]
                first_forecast_date = forecast_df['date'].iloc[0]
                
                # Calculate trend extension (approximate based on slope)
                if len(df_by_date) > 1:
                    trend_slope = (df_by_date['conversions_trend'].iloc[-1] - df_by_date['conversions_trend'].iloc[0]) / len(df_by_date)
                    extended_trend = [last_trend_value + trend_slope * (i + 1) for i in range(len(forecast_df))]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=extended_trend,
                        mode='lines',
                        name='Trend Extension',
                        line=dict(color='#2ca02c', width=2, dash='dot'),
                        opacity=0.5,
                        showlegend=False
                    ))
            
            # Add vertical line to separate historical from forecast
            last_historical_date = historical['date'].max()
            # Get y-axis range for the line
            y_min = min(historical['conversions'].min(), forecast_df['conversions_forecast_lower'].min())
            y_max = max(historical['conversions'].max(), forecast_df['conversions_forecast_upper'].max())
            
            fig.add_shape(
                type="line",
                x0=last_historical_date,
                x1=last_historical_date,
                y0=y_min,
                y1=y_max,
                line=dict(color="gray", width=2, dash="dash"),
                opacity=0.5
            )
            # Add annotation for "Today" label
            fig.add_annotation(
                x=last_historical_date,
                y=y_max,
                text="Today",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                borderpad=3
            )
            
            # Confidence interval (shaded region)
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['conversions_forecast_upper'].tolist() + forecast_df['conversions_forecast_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # ML Forecast (main forecast line)
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['conversions_forecast'],
                mode='lines+markers',
                name='ML Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>' +
                             'Forecast: %{y:,.0f} conversions<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title="Conversions Forecast (Next 7 Days) - Based on Linear Regression Trend",
                xaxis_title="Date",
                yaxis_title="Conversions",
                hovermode='x unified',
                height=450,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                annotations=[
                    dict(
                        text=f"Forecast assumes current {forecast_stats['trend_direction']} trend continues<br>" +
                             f"Confidence: {forecast_stats['confidence']} (R² = {forecast_stats['r_squared']:.2f})",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=-0.15,
                        xanchor="center",
                        font=dict(size=11, color="gray")
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation box
            last_value = historical['conversions'].iloc[-1]
            forecast_change = ((forecast_df['conversions_forecast'].iloc[0] - last_value) / last_value * 100) if last_value > 0 else 0
            
            st.info(
                f"**How to read this forecast:** "
                f"The ML model analyzed your historical data and identified a {forecast_stats['trend_direction']} trend "
                f"(R² = {forecast_stats['r_squared']:.2f}, {forecast_stats['confidence']} confidence). "
                f"The forecast projects this trend forward for the next 7 days. "
                f"Expected change: {forecast_change:+.1f}% from today. "
                f"The shaded area shows the 95% confidence interval - actual values should fall within this range 95% of the time."
            )
            
            # Forecast details table
            st.subheader("Forecast Details")
            forecast_display = forecast_df.copy()
            forecast_display = forecast_display.rename(columns={
                'conversions_forecast': 'Predicted Conversions',
                'conversions_forecast_lower': 'Lower Bound',
                'conversions_forecast_upper': 'Upper Bound'
            })
            forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                forecast_display.style.format({
                    'Predicted Conversions': '{:.0f}',
                    'Lower Bound': '{:.0f}',
                    'Upper Bound': '{:.0f}'
                }),
                use_container_width=True
            )
            
            st.info(f"💡 **Note**: Forecast uses ML-based linear regression. Confidence is {forecast_stats['confidence']} (R² = {forecast_stats['r_squared']:.2f}). "
                   f"Forecasts assume current trends continue and don't account for external factors.")
        else:
            st.warning("⚠️ Not enough data for forecasting. Need at least 7 days of historical data.")
    
    with tab6:
        st.subheader("💰 AI-Powered Budget Optimization")
        st.markdown("**ML-based recommendations** using predictive ROI modeling to optimize budget allocation.")
        
        if optimization_result and optimization_result.get('recommended_allocation'):
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected ROI Improvement",
                    f"+{optimization_result['expected_roi_improvement']:.2f}x",
                    help="Predicted ROI increase from reallocation"
                )
            
            with col2:
                st.metric(
                    "Expected Revenue Increase",
                    f"€{optimization_result['expected_revenue_increase']:,.0f}",
                    help="Additional revenue expected from optimization"
                )
            
            with col3:
                confidence_emoji = "🟢" if optimization_result['confidence'] > 0.7 else "🟡" if optimization_result['confidence'] > 0.4 else "🔴"
                st.metric(
                    "ML Confidence",
                    f"{confidence_emoji} {optimization_result['confidence']:.0%}",
                    help="Confidence in ML predictions"
                )
            
            # Current vs Recommended allocation
            st.subheader("Budget Allocation Comparison")
            
            # Prepare data for visualization
            comparison_data = []
            for channel in optimization_result['recommended_allocation'].keys():
                comparison_data.append({
                    'Channel': channel,
                    'Current Budget (€)': optimization_result['current_allocation'].get(channel, 0),
                    'Recommended Budget (€)': optimization_result['recommended_allocation'][channel],
                    'Change (€)': optimization_result['recommended_allocation'][channel] - optimization_result['current_allocation'].get(channel, 0),
                    'Change (%)': ((optimization_result['recommended_allocation'][channel] - optimization_result['current_allocation'].get(channel, 0)) / 
                                  optimization_result['current_allocation'].get(channel, 1) * 100) if optimization_result['current_allocation'].get(channel, 0) > 0 else 0,
                    'Predicted ROI': optimization_result['predicted_rois'].get(channel, 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Change (€)', key=abs, ascending=False)
            
            # Display table
            st.dataframe(
                comparison_df.style.format({
                    'Current Budget (€)': '€{:,.2f}',
                    'Recommended Budget (€)': '€{:,.2f}',
                    'Change (€)': '€{:,.2f}',
                    'Change (%)': '{:+.1f}%',
                    'Predicted ROI': '{:.2f}x'
                }),
                use_container_width=True,
                height=300
            )
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Current Allocation")
                import plotly.graph_objects as go
                
                fig_current = go.Figure(data=[
                    go.Bar(
                        x=list(optimization_result['current_allocation'].keys()),
                        y=list(optimization_result['current_allocation'].values()),
                        marker_color='#1f77b4',
                        text=[f"€{v:,.0f}" for v in optimization_result['current_allocation'].values()],
                        textposition='auto'
                    )
                ])
                fig_current.update_layout(
                    title="Current Budget Distribution",
                    xaxis_title="Channel",
                    yaxis_title="Budget (€)",
                    height=350
                )
                st.plotly_chart(fig_current, use_container_width=True)
            
            with col2:
                st.subheader("Recommended Allocation")
                
                fig_recommended = go.Figure(data=[
                    go.Bar(
                        x=list(optimization_result['recommended_allocation'].keys()),
                        y=list(optimization_result['recommended_allocation'].values()),
                        marker_color='#2ca02c',
                        text=[f"€{v:,.0f}" for v in optimization_result['recommended_allocation'].values()],
                        textposition='auto'
                    )
                ])
                fig_recommended.update_layout(
                    title="ML-Optimized Budget Distribution",
                    xaxis_title="Channel",
                    yaxis_title="Budget (€)",
                    height=350
                )
                st.plotly_chart(fig_recommended, use_container_width=True)
            
            # Key recommendations
            st.subheader("Key Recommendations")
            for channel in comparison_df['Channel'].head(3):
                row = comparison_df[comparison_df['Channel'] == channel].iloc[0]
                if abs(row['Change (%)']) > 5:
                    direction = "Increase" if row['Change (%)'] > 0 else "Decrease"
                    st.info(
                        f"**{channel}**: {direction} budget by {abs(row['Change (%)']):.0f}% "
                        f"(€{row['Current Budget (€)']:,.0f} → €{row['Recommended Budget (€)']:,.0f}). "
                        f"Predicted ROI: {row['Predicted ROI']:.2f}x"
                    )
            
            st.info(f"💡 **How it works**: Uses ML-based linear regression to predict future ROI for each channel, "
                   f"then optimizes budget allocation to maximize expected return. "
                   f"Confidence: {optimization_result['confidence']:.0%} based on historical trend strength.")
        else:
            st.warning("⚠️ Not enough data for optimization. Need data from multiple channels.")
    
    # Footer
    st.markdown("---")
    st.caption("🤖 Powered by AI | Built with Streamlit | Data processed locally and securely")


if __name__ == "__main__":
    main()

