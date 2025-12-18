# Prototype Development Specification

## Project Overview

**Name**: SME AI-Assisted Analytics Prototype
**Framework**: Streamlit (Python web application)
**Purpose**: Demonstrate feasibility of custom AI analytics solution for SMEs
**Target Users**: Non-technical SME managers (marketing managers, business owners)

### Expected Data Characteristics

This prototype is designed to work with the synthetic data generated from `dataplan.md`, which includes:

- **Dataset size**: 550-600 rows (90 days, 4 channels, 8 campaigns with varying lifecycles)
- **Date range**: 2025-01-01 to 2025-03-31 (3 months)
- **Channels**: Facebook, Google, Instagram, Email (4 channels)
- **Campaigns**: 8 campaigns with realistic start/end dates
- **Embedded anomalies**: 5 types (spending spikes, performance drops, revenue spikes, gradual CTR decline, budget exhaustion)
- **Temporal patterns**: Day-of-week effects, seasonal trends (New Year, Valentine's Day), month-end budget pacing
- **Metrics**: spend, impressions, clicks, conversions, revenue (8 columns)

**Note**: Email channel uses different metric calculation (open rate × click-to-open rate) compared to paid advertising channels.

---

## 1. Requirements Specification

### 1.1 Functional Requirements

#### FR1: Data Import
- **FR1.1**: Accept CSV file upload through web interface
- **FR1.2**: Validate presence of required columns: `date`, `channel`, `campaign`, `spend`, `impressions`, `clicks`, `conversions`, `revenue`
- **FR1.3**: Display error message if required columns missing
- **FR1.4**: Show data preview (first 10 rows) after successful import
- **FR1.5**: Process files up to 10,000 rows without performance degradation

#### FR2: Data Processing
- **FR2.1**: Calculate derived metrics: ROI, CPA (Cost Per Acquisition), CTR (Click-Through Rate), Conversion Rate
  - Note: Email CTR calculated differently (open rate × click-to-open rate) vs. paid ads (impressions → clicks)
- **FR2.2**: Aggregate data by date, channel, campaign as needed
- **FR2.3**: Handle date parsing automatically (support common date formats)
- **FR2.4**: Filter data based on user selections (date range, channels, campaigns)
- **FR2.5**: Handle campaign lifecycle: campaigns have different start/end dates
  - Display only active campaigns in filter dropdowns based on selected date range
  - Show campaign duration in tables/tooltips
  - Handle missing data gracefully when campaigns not yet started or already ended

#### FR3: Anomaly Detection (Core AI Feature #1)
- **FR3.1**: Detect anomalies using Z-score method (threshold: |z| > 2.5)
- **FR3.2**: Identify anomalies in: spend, conversions, revenue, CTR
- **FR3.3**: Categorize anomalies as "high" or "low" for each metric
- **FR3.4**: Track anomaly severity (magnitude of Z-score)
- **FR3.5**: Display anomalies visually on charts (red markers)
- **FR3.6**: List detected anomalies in dedicated section with details
- **FR3.7**: Detect gradual anomalies (trend-based, not just point outliers)
- **FR3.8**: Flag zero-spend periods as potential budget exhaustion
- **FR3.9**: Expected to detect 5 types of anomalies in synthetic data:
  - Spending spikes (technical errors)
  - Performance drops (campaign fatigue)
  - Revenue spikes (viral success)
  - Gradual CTR decline (ad fatigue)
  - Budget exhaustion (zero spend days)

#### FR4: Trend Analysis (Core AI Feature #2)
- **FR4.1**: Calculate 7-day moving average for key metrics
- **FR4.2**: Fit linear regression to identify trend direction
- **FR4.3**: Calculate trend statistics: slope, R², percent change
- **FR4.4**: Classify trend strength: strong (R²>0.7), moderate (0.4-0.7), weak (<0.4)
- **FR4.5**: Display trend lines on time-series charts

#### FR5: Automated Insight Generation (Core AI Feature #3)
- **FR5.1**: Generate 5-10 prioritized plain-language insights
- **FR5.2**: Insight types:
  - Best performing channel (highest ROI)
  - Worst performing channel (ROI <1.0)
  - Anomaly alerts (count and examples - all 5 types)
  - Budget concentration warning (if top 3 campaigns >70% of spend)
  - Performance trend summary (recent vs. older period)
  - Day-of-week patterns (e.g., "Conversions 40% lower on weekends")
  - Month-end budget pacing alerts
  - Seasonal trends (New Year, Valentine's Day effects)
  - Campaign lifecycle notifications (campaigns ending/starting soon)
  - Gradual performance degradation alerts (ad fatigue)
- **FR5.3**: Use emojis for visual categorization (🏆⚠️🔍💰📈📉📅🎯)
- **FR5.4**: Include actionable recommendations in each insight
- **FR5.5**: Display insights in priority order (critical first)

#### FR6: Interactive Visualizations
- **FR6.1**: Time-series chart: metric over time with trend line and anomalies
- **FR6.2**: Channel comparison: bar chart showing ROI by channel
- **FR6.3**: Campaign performance table: sortable, filterable
- **FR6.4**: All charts interactive (zoom, pan, hover tooltips)
- **FR6.5**: Charts update in real-time when filters change

#### FR7: User Interface Controls
- **FR7.1**: Sidebar with filters:
  - Date range slider
  - Channel multi-select dropdown
  - Campaign checkboxes
- **FR7.2**: Main content area with tabs or sections:
  - Insights panel (top)
  - Overview metrics (KPI cards)
  - Visualizations (charts)
  - Data table (detailed view)

#### FR8: Data Export (Optional)
- **FR8.1**: Download filtered data as CSV
- **FR8.2**: Export visualizations as PNG

### 1.2 Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: Load and process 1,000-row dataset in <2 seconds
- **NFR1.2**: Load and process 5,000-row dataset in <5 seconds
- **NFR1.3**: UI remains responsive during processing

#### NFR2: Usability
- **NFR2.1**: Non-technical user can upload data and view insights in <2 minutes
- **NFR2.2**: No coding or SQL knowledge required
- **NFR2.3**: Clear error messages for invalid data
- **NFR2.4**: Intuitive navigation (no training required)

#### NFR3: Accessibility
- **NFR3.1**: Runs on standard laptop (4GB RAM, dual-core CPU)
- **NFR3.2**: No GPU required
- **NFR3.3**: Works on Windows, Mac, Linux

#### NFR4: Privacy/Security
- **NFR4.1**: All processing local (no external API calls)
- **NFR4.2**: No data stored persistently (session-only)
- **NFR4.3**: Data cleared when browser closed

---

## 2. Technical Architecture

### 2.1 Technology Stack

```python
# Core Framework
streamlit==1.28.0          # Web application framework

# Data Processing
pandas==2.1.0              # Data manipulation
numpy==1.26.0              # Numerical operations

# Machine Learning
scikit-learn==1.3.0        # Anomaly detection, linear regression

# Visualization
plotly==5.17.0             # Interactive charts

# Additional
python-dateutil==2.8.2     # Date parsing
```

### 2.2 Project Structure

```
prototype/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── modules/
│   ├── __init__.py
│   ├── data_loader.py     # CSV import and validation
│   ├── data_processor.py  # Metric calculations, aggregations
│   ├── anomaly_detector.py # Z-score anomaly detection
│   ├── trend_analyzer.py   # Moving averages, linear regression
│   ├── insight_generator.py # NLG insight creation
│   └── visualizer.py       # Plotly chart generation
├── utils/
│   ├── __init__.py
│   └── helpers.py          # Utility functions
└── data/
    └── sample_data.csv     # Example dataset (generate using dataplan.md script)
```

### 2.3 Data Flow

```
User uploads CSV
    ↓
data_loader.validate_and_load()
    ↓
data_processor.calculate_metrics()
  - Handle email metrics differently
  - Calculate derived metrics (ROI, CPA, CTR, conversion rate)
    ↓
User applies filters (sidebar)
  - Date range selection
  - Campaign list updates based on lifecycle (active only)
    ↓
data_processor.filter_data()
    ↓
Parallel processing:
├─ anomaly_detector.detect_point_anomalies()
├─ anomaly_detector.detect_gradual_anomalies()
├─ anomaly_detector.detect_zero_spend_anomalies()
├─ trend_analyzer.analyze_trends()
├─ data_processor.aggregate_by_channel()
└─ insight_generator.detect_temporal_patterns()
      ├─ detect_day_of_week_patterns()
      ├─ detect_seasonal_patterns()
      └─ check_campaign_lifecycle()
    ↓
insight_generator.generate_insights()
  - 5-10 insights with priority ranking
  - Include all anomaly types, temporal patterns, lifecycle alerts
    ↓
visualizer.create_charts()
  - Highlight anomalies
  - Show trends
  - Display campaign active periods
    ↓
Display in Streamlit UI
```

---

## 3. Module Specifications

### 3.1 data_loader.py

**Purpose**: Handle CSV import and validation

**Functions**:

```python
def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load CSV from Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        DataFrame with parsed dates and validated columns
    
    Raises:
        ValueError: If required columns missing
    """
    pass

def validate_columns(df: pd.DataFrame) -> bool:
    """
    Check if all required columns present.
    
    Required: date, channel, campaign, spend, impressions, 
              clicks, conversions, revenue
    
    Args:
        df: DataFrame to validate
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    pass
```

### 3.2 data_processor.py

**Purpose**: Calculate derived metrics and aggregate data

**Functions**:

```python
def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived metrics: ROI, CPA, CTR, Conversion Rate.
    
    Formulas:
    - ROI = revenue / spend
    - CPA = spend / conversions
    - CTR = clicks / impressions
    - Conversion Rate = conversions / clicks
    
    Args:
        df: DataFrame with raw metrics
    
    Returns:
        DataFrame with added metric columns
    """
    pass

def filter_data(df: pd.DataFrame, 
                start_date: str, 
                end_date: str,
                channels: list,
                campaigns: list) -> pd.DataFrame:
    """
    Filter data based on user selections.
    
    Args:
        df: Full dataset
        start_date: Start of date range
        end_date: End of date range
        channels: List of selected channels
        campaigns: List of selected campaigns
    
    Returns:
        Filtered DataFrame
    """
    pass

def aggregate_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by channel.
    
    Args:
        df: Transaction-level data
    
    Returns:
        DataFrame with one row per channel, summed metrics
    """
    pass
```

### 3.3 anomaly_detector.py

**Purpose**: Identify statistical anomalies using Z-scores and pattern detection

**Functions**:

```python
def detect_point_anomalies(df: pd.DataFrame, 
                           columns: list = ['spend', 'conversions', 'revenue'],
                           threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect point anomalies using Z-score method.
    
    Z-score formula: z = (x - μ) / σ
    Flag as anomaly if |z| > threshold
    
    Args:
        df: DataFrame with metrics
        columns: Columns to check for anomalies
        threshold: Z-score threshold (default 2.5 = ~1% outliers)
    
    Returns:
        DataFrame with added columns:
        - is_anomaly (bool)
        - anomaly_type (str): description
        - anomaly_severity (float): max |z-score|
    """
    pass

def detect_gradual_anomalies(df: pd.DataFrame, 
                             metric: str = 'ctr',
                             window: int = 7,
                             decline_threshold: float = 0.05) -> pd.DataFrame:
    """
    Detect gradual performance degradation over time.
    
    Uses rolling window comparison to identify sustained declines.
    
    Args:
        df: DataFrame sorted by date
        metric: Column to analyze (e.g., 'ctr', 'conversion_rate')
        window: Days for rolling comparison
        decline_threshold: Minimum decline rate to flag (e.g., 0.05 = 5%)
    
    Returns:
        DataFrame with gradual_anomaly flags and decline percentages
    """
    pass

def detect_zero_spend_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect budget exhaustion (unexpected zero-spend periods).
    
    Flags days with zero spend when campaign should be active.
    
    Args:
        df: DataFrame with spend column
    
    Returns:
        DataFrame with budget_exhaustion flags
    """
    pass

def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Summarize detected anomalies across all types.
    
    Returns:
        {
            'count': int,
            'most_recent': list of dicts with details,
            'by_type': dict of counts by anomaly type,
            'by_category': {
                'point': int,
                'gradual': int,
                'budget': int
            }
        }
    """
    pass
```

### 3.4 trend_analyzer.py

**Purpose**: Calculate trends using moving averages and regression

**Functions**:

```python
def analyze_trends(df: pd.DataFrame, 
                   metric: str = 'conversions',
                   window: int = 7) -> tuple:
    """
    Calculate moving average and fit linear trend.
    
    Args:
        df: DataFrame sorted by date
        metric: Column name to analyze
        window: Days for moving average
    
    Returns:
        Tuple of (df_with_trends, trend_stats)
        
        df_with_trends has added columns:
        - {metric}_ma{window}: moving average
        - {metric}_trend: linear trend line
        
        trend_stats = {
            'direction': 'increasing' or 'decreasing',
            'slope': float,
            'percent_change': float,
            'r_squared': float,
            'strength': 'strong' | 'moderate' | 'weak'
        }
    """
    pass
```

### 3.5 insight_generator.py

**Purpose**: Generate plain-language insights from analysis results

**Functions**:

```python
def generate_insights(df: pd.DataFrame,
                      anomaly_summary: dict,
                      trend_stats: dict,
                      date_range: tuple) -> list:
    """
    Generate prioritized natural language insights.
    
    Args:
        df: Processed data with metrics
        anomaly_summary: From anomaly_detector.get_anomaly_summary()
        trend_stats: From trend_analyzer.analyze_trends()
        date_range: (start_date, end_date) for context
    
    Returns:
        List of insight dicts:
        [
            {
                'text': str (markdown-formatted),
                'priority': int (1=highest),
                'type': 'success' | 'warning' | 'info',
                'category': str (e.g., 'performance', 'anomaly', 'temporal')
            },
            ...
        ]
        
    Insight types generated:
    1. Best performing channel (priority 1)
    2. Worst performing channel if ROI<1 (priority 2)
    3. Anomaly alerts - all 5 types (priority 2)
    4. Budget concentration warning (priority 3)
    5. Trend summary (priority 3)
    6. Day-of-week patterns (priority 3)
    7. Seasonal effects (priority 3)
    8. Campaign lifecycle alerts (priority 3)
    9. Gradual degradation warnings (priority 2)
    10. Month-end budget pacing (priority 3)
    """
    pass

def detect_day_of_week_patterns(df: pd.DataFrame) -> dict:
    """
    Analyze performance by day of week.
    
    Returns:
        {
            'weekend_effect': float (percent change),
            'best_day': str,
            'worst_day': str,
            'insights': list of strings
        }
    """
    pass

def detect_seasonal_patterns(df: pd.DataFrame) -> list:
    """
    Identify seasonal trends (New Year, Valentine's, month-end).
    
    Returns:
        List of seasonal insight strings
    """
    pass

def check_campaign_lifecycle(df: pd.DataFrame, current_date: str) -> list:
    """
    Check for campaigns starting/ending soon.
    
    Returns:
        List of campaign lifecycle insight strings
    """
    pass
```

### 3.6 visualizer.py

**Purpose**: Create interactive Plotly charts

**Functions**:

```python
def create_timeseries_chart(df: pd.DataFrame,
                           metric: str,
                           show_anomalies: bool = True,
                           show_trend: bool = True) -> go.Figure:
    """
    Create interactive time-series chart.
    
    Args:
        df: Data with date, metric, optional anomaly flags and trend
        metric: Column name to plot
        show_anomalies: Highlight anomalies with red markers
        show_trend: Show trend line and moving average
    
    Returns:
        Plotly Figure object
    """
    pass

def create_channel_comparison_chart(channel_metrics: pd.DataFrame) -> go.Figure:
    """
    Create bar chart comparing channels.
    
    Args:
        channel_metrics: DataFrame with channel, roi columns
    
    Returns:
        Plotly Figure object with horizontal bar chart
    """
    pass

def create_campaign_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted campaign performance table.
    
    Args:
        df: Campaign-level aggregated data
    
    Returns:
        Styled DataFrame suitable for st.dataframe()
    """
    pass
```

---

## 4. User Interface Specification

### 4.1 Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│  SIDEBAR                │  MAIN CONTENT                  │
│                         │                                │
│  📁 File Upload         │  🔍 AI-Assisted Analytics      │
│  [Upload CSV]           │                                │
│                         │  ┌─────────────────────────┐  │
│  📅 Date Range          │  │  INSIGHTS PANEL         │  │
│  [Slider]               │  │  🏆 Top performer...    │  │
│                         │  │  ⚠️ Attention needed... │  │
│  📊 Filters             │  │  🔍 Anomalies detected..│  │
│  ☐ Facebook             │  └─────────────────────────┘  │
│  ☐ Google               │                                │
│  ☐ Instagram            │  ┌─────────────────────────┐  │
│  ☐ Email                │  │  KEY METRICS            │  │
│                         │  │  Total Spend: €X        │  │
│  📈 Campaigns           │  │  Total Revenue: €Y      │  │
│  ☐ Campaign 1           │  │  Overall ROI: Z.ZZx     │  │
│  ☐ Campaign 2           │  └─────────────────────────┘  │
│  ☐ Campaign 3           │                                │
│                         │  📊 VISUALIZATIONS             │
│                         │  [Conversions Over Time Chart] │
│                         │  [Channel ROI Comparison]      │
│                         │                                │
│                         │  📋 DATA TABLE                 │
│                         │  [Campaign Performance Table]  │
└─────────────────────────────────────────────────────────┘
```

### 4.2 UI Components (Streamlit Code)

```python
# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.header("📅 Date Range")
    date_range = st.date_input("Select range", [start_date, end_date])
    
    st.header("📊 Filters")
    selected_channels = st.multiselect("Channels", all_channels)
    selected_campaigns = st.multiselect("Campaigns", all_campaigns)

# Main content
st.title("🔍 AI-Assisted Marketing Analytics")

# Insights panel
st.header("💡 Key Insights")
for insight in insights:
    if insight['type'] == 'success':
        st.success(insight['text'])
    elif insight['type'] == 'warning':
        st.warning(insight['text'])
    else:
        st.info(insight['text'])

# Key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Spend", f"€{total_spend:,.0f}")
with col2:
    st.metric("Total Revenue", f"€{total_revenue:,.0f}")
with col3:
    st.metric("Overall ROI", f"{overall_roi:.2f}x")

# Visualizations
st.header("📊 Performance Trends")
st.plotly_chart(conversion_chart, use_container_width=True)

st.header("🏆 Channel Comparison")
st.plotly_chart(channel_chart, use_container_width=True)

# Data table
st.header("📋 Campaign Details")
st.dataframe(campaign_table, use_container_width=True)
```

---

## 5. Development Plan

### Phase 1: Setup and Basic Structure (2-4 hours)
- [ ] Create project directory structure
- [ ] Set up virtual environment
- [ ] Install dependencies (requirements.txt)
- [ ] Create basic Streamlit app.py with layout
- [ ] Test that Streamlit runs

### Phase 2: Data Loading Module (2-3 hours)
- [ ] Implement data_loader.py
- [ ] Add CSV validation
- [ ] Create sample test data
- [ ] Test file upload in Streamlit
- [ ] Add error handling

### Phase 3: Data Processing Module (3-4 hours)
- [ ] Implement data_processor.py
- [ ] Add metric calculations (ROI, CPA, CTR, conversion rate)
- [ ] Add filtering functions
- [ ] Add aggregation functions
- [ ] Test with sample data

### Phase 4: Anomaly Detection (6-8 hours)
- [ ] Implement anomaly_detector.py
- [ ] Code Z-score calculation (point anomalies)
- [ ] Implement gradual anomaly detection (rolling window)
- [ ] Implement zero-spend anomaly detection
- [ ] Test with all 5 known anomaly types from synthetic data
- [ ] Integrate with main app
- [ ] Add visual highlighting for each anomaly type

### Phase 5: Trend Analysis (3-4 hours)
- [ ] Implement trend_analyzer.py
- [ ] Add moving average calculation
- [ ] Add linear regression
- [ ] Calculate trend statistics
- [ ] Test with time-series data

### Phase 6: Insight Generation (6-8 hours)
- [ ] Implement insight_generator.py
- [ ] Create insight templates for all 10 types
- [ ] Implement detect_day_of_week_patterns()
- [ ] Implement detect_seasonal_patterns()
- [ ] Implement check_campaign_lifecycle()
- [ ] Add priority logic (1-3 with proper ordering)
- [ ] Test with synthetic data scenarios
- [ ] Verify all 5 anomalies generate appropriate insights
- [ ] Refine language and formatting for non-technical users

### Phase 7: Visualizations (5-6 hours)
- [ ] Implement visualizer.py
- [ ] Create time-series chart
- [ ] Create channel comparison chart
- [ ] Create campaign table
- [ ] Add interactivity
- [ ] Style and polish

### Phase 8: UI Integration (4-5 hours)
- [ ] Build complete Streamlit UI
- [ ] Add sidebar filters
- [ ] Connect all components
- [ ] Add state management
- [ ] Test user workflow

### Phase 9: Testing and Refinement (4-6 hours)
- [ ] Test with multiple datasets
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] UI/UX refinements
- [ ] Documentation

### Phase 10: Documentation (2-3 hours)
- [ ] Write README.md
- [ ] Add code comments
- [ ] Create user guide
- [ ] Document deployment options
- [ ] Reference dataplan.md for test data generation

**Total Estimated Time: 40-55 hours**

*Note: Additional 5 hours added for enhanced anomaly detection (3 types → 5 types) and temporal pattern analysis.*

---

## 6. Testing Strategy

### 6.1 Unit Tests

Test each module independently:

```python
# test_anomaly_detector.py
def test_detect_anomalies_with_known_outlier():
    # Create dataset with one known outlier
    # Assert outlier is detected
    pass

def test_no_anomalies_in_normal_data():
    # Create dataset with no outliers
    # Assert no anomalies detected
    pass
```

### 6.2 Integration Tests

Test complete workflows:

```python
# test_full_workflow.py
def test_upload_to_insights():
    # Load sample CSV
    # Process through all modules
    # Assert insights generated
    pass
```

### 6.3 Manual Testing Checklist

**Data Upload & Validation**:
- [ ] Upload valid CSV file (generated from dataplan.md)
- [ ] Upload invalid CSV (missing columns)
- [ ] Upload very large file (10,000 rows)

**Filtering & Navigation**:
- [ ] Apply date filter
- [ ] Apply channel filter (should update campaign list)
- [ ] Apply campaign filter
- [ ] Filter shows only active campaigns for selected date range

**Anomaly Detection (5 types)**:
- [ ] Verify spending spike detected (Feb 15, Google_Search_Brand)
- [ ] Verify performance drop detected (Mar 10-15, FB_Retargeting)
- [ ] Verify revenue spike detected (Feb 23, Email_Newsletter)
- [ ] Verify gradual CTR decline detected (Feb, IG_Influencer)
- [ ] Verify budget exhaustion detected (Mar 22-23, Google_Display)

**Temporal Patterns**:
- [ ] Verify weekend effect detected (~40% lower conversions)
- [ ] Verify day-of-week patterns identified
- [ ] Verify seasonal trends noted (New Year, Valentine's Day)
- [ ] Verify month-end budget pacing detected

**Campaign Lifecycle**:
- [ ] Campaign dates displayed correctly
- [ ] Ended campaigns flagged appropriately
- [ ] Campaign lifecycle insights generated

**Performance & UX**:
- [ ] Verify trends calculated correctly
- [ ] Verify insights make sense and are actionable
- [ ] Check all charts render
- [ ] Test chart interactivity
- [ ] Verify performance (<5 sec for 5K rows)
- [ ] Email CTR calculated correctly (different from paid ads)

---

## 7. Deployment Options

### Option 1: Local Development
```bash
# Clone/download project
cd prototype
pip install -r requirements.txt
streamlit run app.py
# Opens in browser at localhost:8501
```

### Option 2: Streamlit Community Cloud (Free)
1. Push code to GitHub repository
2. Go to share.streamlit.io
3. Connect GitHub repo
4. Deploy (automatic)
5. Get public URL

### Option 3: Docker Container
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 8. Success Criteria

### Functional Success
- ✅ All 8 functional requirements implemented (including FR2.5 campaign lifecycle)
- ✅ All 3 AI features working correctly with enhanced capabilities
- ✅ 5 anomaly types detected successfully:
  - Point anomalies (spending spikes, revenue spikes, performance drops)
  - Gradual anomalies (CTR decline over time)
  - Budget anomalies (zero-spend periods)
- ✅ Temporal pattern detection working (day-of-week, seasonal, month-end)
- ✅ Campaign lifecycle handling functional
- ✅ No critical bugs

### Performance Success
- ✅ Processes 550-600 rows (synthetic dataset) in <1 second
- ✅ Processes 1,000 rows in <2 seconds
- ✅ Processes 5,000 rows in <5 seconds
- ✅ UI remains responsive during all operations

### Usability Success
- ✅ Non-technical user can complete workflow in <2 minutes
- ✅ No training required
- ✅ Clear error messages for invalid data
- ✅ Campaign filters adapt to date range selection
- ✅ Insights are actionable and easy to understand

### Data Compatibility Success
- ✅ Successfully processes synthetic data from dataplan.md
- ✅ Detects all 5 embedded anomalies correctly
- ✅ Identifies temporal patterns (weekends, New Year, Valentine's Day)
- ✅ Handles campaign lifecycle (campaigns with start/end dates)
- ✅ Correctly calculates email metrics vs. paid ad metrics

### Academic Success
- ✅ Demonstrates feasibility of custom development
- ✅ Validates cost estimates (35-50 hours)
- ✅ Proves value delivery (95% time savings)
- ✅ Shows AI features work as designed
- ✅ Validates synthetic data generation approach

---

## 9. Known Limitations & Future Enhancements

### Current Limitations
- Single-user (no multi-tenancy)
- No persistent storage
- No user authentication
- English language only
- Basic NLG (template-based)

### Potential Enhancements (Not Required)
- Forecasting feature (Holt-Winters)
- Customer segmentation (K-means)
- Export to PDF report
- Multi-language support
- Database connectivity
- User authentication

---

## 10. Reference Implementation Hints

### Key Libraries Usage

**Streamlit basics**:
```python
import streamlit as st

# File upload
uploaded_file = st.file_uploader("Choose CSV", type='csv')

# Filters
channels = st.multiselect("Channels", options=['FB', 'Google'])

# Display
st.write("Hello")
st.dataframe(df)
st.plotly_chart(fig)
```

**Pandas processing**:
```python
import pandas as pd

# Load CSV
df = pd.read_csv(uploaded_file)

# Calculate metrics
df['roi'] = df['revenue'] / df['spend']

# Filter
filtered = df[df['channel'].isin(selected_channels)]

# Aggregate
by_channel = df.groupby('channel').agg({'spend': 'sum', 'revenue': 'sum'})
```

**Scikit-learn anomaly detection**:
```python
from sklearn.preprocessing import StandardScaler

# Z-score calculation
mean = df['spend'].mean()
std = df['spend'].std()
df['z_score'] = (df['spend'] - mean) / std
df['is_anomaly'] = abs(df['z_score']) > 2.5
```

**Plotly visualization**:
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['conversions'], mode='lines+markers'))
fig.update_layout(title='Conversions Over Time')
```

---

## 11. Alignment with Data Plan

This prototype specification is fully aligned with `dataplan.md`:

### Data Compatibility Matrix

| Data Plan Feature | Prototype Capability | Status |
|-------------------|---------------------|---------|
| 550-600 row dataset | Handles up to 10,000 rows | ✅ Compatible |
| 8 campaigns with lifecycle | Campaign filter + lifecycle handling (FR2.5) | ✅ Implemented |
| 5 anomaly types embedded | 5 anomaly detection methods (FR3.7-FR3.9) | ✅ Full coverage |
| Day-of-week patterns | detect_day_of_week_patterns() | ✅ Implemented |
| Seasonal trends | detect_seasonal_patterns() | ✅ Implemented |
| Month-end pacing | Temporal pattern insights | ✅ Implemented |
| Email metrics (different calc) | Special handling in FR2.1 | ✅ Documented |
| Campaign start/end dates | Campaign lifecycle checks | ✅ Implemented |

### Expected Anomaly Detection Results

Using synthetic data from dataplan.md, the prototype should detect:

1. **Feb 15, 2025** - Google_Search_Brand spending spike (~€700, 3.5x normal) → Z-score >3.0
2. **Mar 10-15, 2025** - FB_Retargeting performance drop (70% below normal) → Z-score <-2.5
3. **Feb 23, 2025** - Email_Newsletter revenue spike (4x normal) → Z-score >3.0
4. **Feb 1-28, 2025** - IG_Influencer gradual CTR decline (8% over month) → Trend detection
5. **Mar 22-23, 2025** - Google_Display budget exhaustion (€0 spend) → Zero-spend flag

### Insight Generation Examples

Expected insights from synthetic data:

- 🏆 "**Google Ads** is your top performer with **2.8x ROI** - significantly outperforming other channels"
- ⚠️ "**FB_Retargeting** conversions dropped **70%** during Mar 10-15. Consider refreshing ad creative."
- 🔍 "**5 anomalies detected** in your data - 2 require immediate attention"
- 📅 "Conversions are **40% lower on weekends** - consider adjusting budget allocation"
- 🎯 "**IG_Story_Promo** campaign ended on Feb 14 - performance data complete"

---

## 12. Quick Start Guide

### Step 1: Generate Test Data
```bash
cd prototype
python generate_synthetic_data.py --visualize
# Generates: synthetic_marketing_data.csv
```

### Step 2: Run Prototype
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Step 3: Validate AI Features
1. Upload `synthetic_marketing_data.csv`
2. Verify 5 anomalies detected in insights panel
3. Check charts show anomaly markers (red dots)
4. Verify day-of-week and seasonal patterns mentioned
5. Confirm campaign lifecycle properly displayed

---

This specification provides everything needed to build the prototype with full compatibility for the enhanced synthetic dataset! 🚀
