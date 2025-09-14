
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .insight-box {
        background: #2d3748;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        color: #ffffff;
    }
    .insight-box h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .insight-box p {
        color: #e2e8f0;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Marketing Intelligence Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive analysis of marketing performance and business impact**")

@st.cache_data
def load_and_process_data():
    """
    Load and process all marketing and business data.
    Returns cleaned and processed datasets for dashboard visualization.
    """
    try:
        # Load datasets
        business_df = pd.read_csv('datasets/business.csv')
        facebook_df = pd.read_csv('datasets/Facebook.csv')
        google_df = pd.read_csv('datasets/Google.csv')
        tiktok_df = pd.read_csv('datasets/TikTok.csv')
        
        # Clean column names
        def clean_columns(df):
            df = df.copy()
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('#', '').str.replace('-', '_')
            return df
        
        business_df = clean_columns(business_df)
        facebook_df = clean_columns(facebook_df)
        google_df = clean_columns(google_df)
        tiktok_df = clean_columns(tiktok_df)
        
        # Fix business column names
        business_df = business_df.rename(columns={
            '_of_orders': 'orders',
            '_of_new_orders': 'new_orders'
        })
        
        # Convert date columns
        for df in [business_df, facebook_df, google_df, tiktok_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Add channel labels
        facebook_df['channel'] = 'Facebook'
        google_df['channel'] = 'Google'
        tiktok_df['channel'] = 'TikTok'
        
        # Standardize marketing data columns
        marketing_cols = ['date', 'tactic', 'state', 'campaign', 'impressions', 'clicks', 'spend', 'attributed_revenue', 'channel']
        
        def standardize_marketing_df(df):
            df = df.copy()
            # Rename impression to impressions for consistency
            if 'impression' in df.columns:
                df = df.rename(columns={'impression': 'impressions'})
            # Select and reorder columns - include impressions
            available_cols = [col for col in marketing_cols if col in df.columns]
            # Make sure impressions is included if it exists
            if 'impressions' in df.columns and 'impressions' not in available_cols:
                available_cols.append('impressions')
            return df[available_cols]
        
        facebook_df = standardize_marketing_df(facebook_df)
        google_df = standardize_marketing_df(google_df)
        tiktok_df = standardize_marketing_df(tiktok_df)
        
        # Combine all marketing data
        marketing_df = pd.concat([facebook_df, google_df, tiktok_df], ignore_index=True)
        
        # Convert numeric columns
        numeric_cols = ['impressions', 'clicks', 'spend', 'attributed_revenue']
        for col in numeric_cols:
            if col in marketing_df.columns:
                marketing_df[col] = pd.to_numeric(marketing_df[col], errors='coerce').fillna(0)
        
        # Convert business numeric columns
        business_numeric_cols = ['orders', 'new_orders', 'new_customers', 'total_revenue', 'gross_profit', 'cogs']
        for col in business_numeric_cols:
            if col in business_df.columns:
                business_df[col] = pd.to_numeric(business_df[col], errors='coerce').fillna(0)
        
        # Calculate marketing metrics
        marketing_df['ctr'] = np.where(
            marketing_df['impressions'] > 0,
            marketing_df['clicks'] / marketing_df['impressions'],
            0
        )
        marketing_df['cpc'] = np.where(
            marketing_df['clicks'] > 0,
            marketing_df['spend'] / marketing_df['clicks'],
            np.nan
        )
        marketing_df['cpm'] = np.where(
            marketing_df['impressions'] > 0,
            marketing_df['spend'] / (marketing_df['impressions'] / 1000),
            np.nan
        )
        marketing_df['roas'] = np.where(
            marketing_df['spend'] > 0,
            marketing_df['attributed_revenue'] / marketing_df['spend'],
            np.nan
        )
        
        # Calculate business metrics
        business_df['aov'] = np.where(
            business_df['orders'] > 0,
            business_df['total_revenue'] / business_df['orders'],
            np.nan
        )
        business_df['gross_margin'] = np.where(
            business_df['total_revenue'] > 0,
            business_df['gross_profit'] / business_df['total_revenue'],
            np.nan
        )
        
        # Create daily aggregated marketing data
        daily_marketing = marketing_df.groupby(['date', 'channel']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        daily_marketing['roas'] = np.where(
            daily_marketing['spend'] > 0,
            daily_marketing['attributed_revenue'] / daily_marketing['spend'],
            np.nan
        )
        
        # Create total daily marketing data
        daily_total_marketing = daily_marketing.groupby('date').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        daily_total_marketing['roas'] = np.where(
            daily_total_marketing['spend'] > 0,
            daily_total_marketing['attributed_revenue'] / daily_total_marketing['spend'],
            np.nan
        )
        
        # Merge business and marketing data
        daily_combined = pd.merge(
            business_df, 
            daily_total_marketing, 
            on='date', 
            how='left'
        ).fillna(0)
        
        # Calculate combined metrics
        daily_combined['mer'] = np.where(
            daily_combined['spend'] > 0,
            daily_combined['total_revenue'] / daily_combined['spend'],
            np.nan
        )
        daily_combined['cac'] = np.where(
            daily_combined['new_customers'] > 0,
            daily_combined['spend'] / daily_combined['new_customers'],
            np.nan
        )
        daily_combined['attribution_rate'] = np.where(
            daily_combined['total_revenue'] > 0,
            daily_combined['attributed_revenue'] / daily_combined['total_revenue'],
            np.nan
        )
        
        # Campaign performance analysis
        campaign_performance = marketing_df.groupby(['channel', 'tactic', 'state', 'campaign']).agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        campaign_performance['ctr'] = np.where(
            campaign_performance['impressions'] > 0,
            campaign_performance['clicks'] / campaign_performance['impressions'],
            0
        )
        campaign_performance['cpc'] = np.where(
            campaign_performance['clicks'] > 0,
            campaign_performance['spend'] / campaign_performance['clicks'],
            np.nan
        )
        campaign_performance['roas'] = np.where(
            campaign_performance['spend'] > 0,
            campaign_performance['attributed_revenue'] / campaign_performance['spend'],
            np.nan
        )
        
        return {
            'marketing': marketing_df,
            'business': business_df,
            'daily_marketing': daily_marketing,
            'daily_combined': daily_combined,
            'campaign_performance': campaign_performance
        }
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def format_currency(value):
    """Format value as currency"""
    if pd.isna(value) or value == 0:
        return "$0"
    return f"${value:,.0f}"

def format_percentage(value):
    """Format value as percentage"""
    if pd.isna(value):
        return "0%"
    return f"{value:.1%}"

def format_ratio(value):
    """Format value as ratio"""
    if pd.isna(value):
        return "0.0x"
    return f"{value:.2f}x"

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" and delta > 0 else "red" if delta_color == "inverse" and delta > 0 else "gray"
        delta_html = f'<div style="font-size: 0.8rem; color: {color};">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

# Sidebar filters
st.sidebar.header("🔍 Filters")

def main():
    # Load data
    data = load_and_process_data()
    if data is None:
        st.error("Failed to load data. Please check your dataset files.")
        return
    
    # Date range filter
    min_date = data['daily_combined']['date'].min()
    max_date = data['daily_combined']['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date
    
    # Channel filter
    available_channels = data['marketing']['channel'].unique()
    selected_channels = st.sidebar.multiselect(
        "Select Channels",
        options=available_channels,
        default=available_channels
    )
    
    # Additional interactive filters
    st.sidebar.subheader("Advanced Filters")
    
    # State filter
    available_states = data['marketing']['state'].unique()
    selected_states = st.sidebar.multiselect(
        "Select States",
        options=available_states,
        default=available_states
    )
    
    # Tactic filter
    available_tactics = data['marketing']['tactic'].unique()
    selected_tactics = st.sidebar.multiselect(
        "Select Tactics",
        options=available_tactics,
        default=available_tactics
    )
    
    # Performance threshold filters
    st.sidebar.subheader("Performance Filters")
    
    min_roas = st.sidebar.slider(
        "Minimum ROAS",
        min_value=0.0,
        max_value=float(data['marketing']['roas'].max()),
        value=0.0,
        step=0.1
    )
    
    min_spend = st.sidebar.slider(
        "Minimum Spend ($)",
        min_value=0.0,
        max_value=float(data['marketing']['spend'].max()),
        value=0.0,
        step=1000.0
    )
    
    # Apply filters
    filtered_marketing = data['marketing'][
        (data['marketing']['date'] >= start_date) & 
        (data['marketing']['date'] <= end_date) &
        (data['marketing']['channel'].isin(selected_channels)) &
        (data['marketing']['state'].isin(selected_states)) &
        (data['marketing']['tactic'].isin(selected_tactics)) &
        (data['marketing']['roas'] >= min_roas) &
        (data['marketing']['spend'] >= min_spend)
    ].copy()
    
    filtered_business = data['business'][
        (data['business']['date'] >= start_date) & 
        (data['business']['date'] <= end_date)
    ].copy()
    
    # Calculate filtered metrics
    total_spend = filtered_marketing['spend'].sum()
    total_revenue = filtered_business['total_revenue'].sum()
    total_attributed_revenue = filtered_marketing['attributed_revenue'].sum()
    total_orders = filtered_business['orders'].sum()
    total_new_customers = filtered_business['new_customers'].sum()
    
    # Filter Summary
    st.header("🔍 Filter Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Date Range", f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    with col2:
        st.metric("Channels Selected", len(selected_channels))
    
    with col3:
        st.metric("States Selected", len(selected_states))
    
    with col4:
        st.metric("Tactics Selected", len(selected_tactics))
    
    # Data summary
    st.info(f"📊 Showing data for {filtered_marketing.shape[0]} marketing records and {filtered_business.shape[0]} business days")
    
    # Key Performance Indicators
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Spend",
            format_currency(total_spend)
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Total Revenue",
            format_currency(total_revenue)
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Attributed Revenue",
            format_currency(total_attributed_revenue)
        ), unsafe_allow_html=True)
    
    with col4:
        overall_roas = total_attributed_revenue / total_spend if total_spend > 0 else 0
        st.markdown(create_metric_card(
            "Overall ROAS",
            format_ratio(overall_roas)
        ), unsafe_allow_html=True)
    
    with col5:
        mer = total_revenue / total_spend if total_spend > 0 else 0
        st.markdown(create_metric_card(
            "Marketing Efficiency Ratio",
            format_ratio(mer)
        ), unsafe_allow_html=True)
    
    with col6:
        cac = total_spend / total_new_customers if total_new_customers > 0 else 0
        st.markdown(create_metric_card(
            "Customer Acquisition Cost",
            format_currency(cac)
        ), unsafe_allow_html=True)
    
    # Revenue vs Spend Trend
    st.header("💰 Revenue vs Marketing Spend Trend")
    
    daily_filtered = data['daily_combined'][
        (data['daily_combined']['date'] >= start_date) & 
        (data['daily_combined']['date'] <= end_date)
    ].copy()
    
    fig_revenue_spend = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add spend bars
    fig_revenue_spend.add_trace(
        go.Bar(
            x=daily_filtered['date'],
            y=daily_filtered['spend'],
            name='Marketing Spend',
            marker_color='#3b82f6',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Add revenue lines
    fig_revenue_spend.add_trace(
        go.Scatter(
            x=daily_filtered['date'],
            y=daily_filtered['total_revenue'],
            name='Total Revenue',
            mode='lines+markers',
            line=dict(color='#10b981', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    fig_revenue_spend.add_trace(
        go.Scatter(
            x=daily_filtered['date'],
            y=daily_filtered['attributed_revenue'],
            name='Attributed Revenue',
            mode='lines+markers',
            line=dict(color='#f59e0b', width=3),
            marker=dict(size=6)
        ),
        secondary_y=True
    )
    
    fig_revenue_spend.update_layout(
        title="Daily Marketing Spend vs Revenue",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    fig_revenue_spend.update_yaxes(title_text="Marketing Spend ($)", secondary_y=False)
    fig_revenue_spend.update_yaxes(title_text="Revenue ($)", secondary_y=True)
    
    st.plotly_chart(fig_revenue_spend, use_container_width=True)
    
    # Channel Performance Analysis
    st.header("📊 Channel Performance Analysis")
    
    # Channel metrics
    channel_metrics = filtered_marketing.groupby('channel').agg({
        'spend': 'sum',
        'attributed_revenue': 'sum',
        'impressions': 'sum',
        'clicks': 'sum'
    }).reset_index()
    
    channel_metrics['roas'] = channel_metrics['attributed_revenue'] / channel_metrics['spend']
    channel_metrics['ctr'] = channel_metrics['clicks'] / channel_metrics['impressions']
    channel_metrics['cpm'] = channel_metrics['spend'] / (channel_metrics['impressions'] / 1000)
    
    # Interactive channel selection
    st.subheader("Interactive Channel Analysis")
    selected_channel = st.selectbox(
        "Select a channel to drill down:",
        options=channel_metrics['channel'].tolist(),
        index=0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROAS by Channel
        fig_roas = px.bar(
            channel_metrics,
            x='channel',
            y='roas',
            title='ROAS by Channel',
            color='roas',
            color_continuous_scale='RdYlGn'
        )
        fig_roas.update_layout(height=400)
        st.plotly_chart(fig_roas, use_container_width=True)
    
    with col2:
        # Spend Distribution
        fig_spend = px.pie(
            channel_metrics,
            values='spend',
            names='channel',
            title='Spend Distribution by Channel'
        )
        fig_spend.update_layout(height=400)
        st.plotly_chart(fig_spend, use_container_width=True)
    
    # Channel drill-down
    if selected_channel:
        st.subheader(f"📈 {selected_channel} Channel Deep Dive")
        
        # Filter data for selected channel
        channel_data = filtered_marketing[filtered_marketing['channel'] == selected_channel]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            channel_spend = channel_data['spend'].sum()
            st.metric(f"{selected_channel} Total Spend", format_currency(channel_spend))
        
        with col2:
            channel_revenue = channel_data['attributed_revenue'].sum()
            st.metric(f"{selected_channel} Attributed Revenue", format_currency(channel_revenue))
        
        with col3:
            channel_roas = channel_revenue / channel_spend if channel_spend > 0 else 0
            st.metric(f"{selected_channel} ROAS", format_ratio(channel_roas))
        
        # Channel performance over time
        channel_daily = channel_data.groupby('date').agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'impressions': 'sum',
            'clicks': 'sum'
        }).reset_index()
        
        channel_daily['roas'] = channel_daily['attributed_revenue'] / channel_daily['spend']
        
        fig_channel_trend = go.Figure()
        fig_channel_trend.add_trace(go.Scatter(
            x=channel_daily['date'],
            y=channel_daily['spend'],
            name='Spend',
            mode='lines+markers',
            line=dict(color='#3b82f6')
        ))
        fig_channel_trend.add_trace(go.Scatter(
            x=channel_daily['date'],
            y=channel_daily['attributed_revenue'],
            name='Attributed Revenue',
            mode='lines+markers',
            line=dict(color='#10b981')
        ))
        fig_channel_trend.update_layout(
            title=f'{selected_channel} Performance Over Time',
            height=400
        )
        st.plotly_chart(fig_channel_trend, use_container_width=True)
        
        # Top campaigns for selected channel
        st.subheader(f"Top Campaigns in {selected_channel}")
        channel_campaigns = channel_data.groupby(['tactic', 'state', 'campaign']).agg({
            'spend': 'sum',
            'attributed_revenue': 'sum',
            'roas': 'mean'
        }).reset_index().sort_values('roas', ascending=False)
        
        channel_campaigns['spend'] = channel_campaigns['spend'].apply(format_currency)
        channel_campaigns['attributed_revenue'] = channel_campaigns['attributed_revenue'].apply(format_currency)
        channel_campaigns['roas'] = channel_campaigns['roas'].apply(format_ratio)
        
        st.dataframe(
            channel_campaigns.head(10),
            use_container_width=True
        )
    
    # Channel Performance Table
    st.subheader("Channel Performance Summary")
    
    display_metrics = channel_metrics.copy()
    display_metrics['spend'] = display_metrics['spend'].apply(format_currency)
    display_metrics['attributed_revenue'] = display_metrics['attributed_revenue'].apply(format_currency)
    display_metrics['roas'] = display_metrics['roas'].apply(format_ratio)
    display_metrics['ctr'] = display_metrics['ctr'].apply(format_percentage)
    display_metrics['cpm'] = display_metrics['cpm'].apply(format_currency)
    
    st.dataframe(
        display_metrics[['channel', 'spend', 'attributed_revenue', 'roas', 'ctr', 'cpm']],
        use_container_width=True
    )
    
    # Campaign Performance
    st.header("🎯 Campaign Performance Analysis")
    
    # Interactive campaign analysis
    st.subheader("Interactive Campaign Analysis")
    
    # Campaign filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        campaign_sort_by = st.selectbox(
            "Sort campaigns by:",
            options=['roas', 'spend', 'attributed_revenue', 'clicks'],
            index=0
        )
    
    with col2:
        campaign_limit = st.slider(
            "Number of campaigns to show:",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
    
    with col3:
        min_campaign_spend = st.slider(
            "Minimum campaign spend ($):",
            min_value=0.0,
            max_value=float(data['campaign_performance']['spend'].max()),
            value=0.0,
            step=1000.0
        )
    
    # Filter and sort campaigns
    filtered_campaigns = data['campaign_performance'][
        data['campaign_performance']['spend'] >= min_campaign_spend
    ].nlargest(campaign_limit, campaign_sort_by)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Top {campaign_limit} Campaigns by {campaign_sort_by.title()}")
        campaigns_display = filtered_campaigns[['channel', 'campaign', 'spend', 'attributed_revenue', 'roas']].copy()
        campaigns_display['spend'] = campaigns_display['spend'].apply(format_currency)
        campaigns_display['attributed_revenue'] = campaigns_display['attributed_revenue'].apply(format_currency)
        campaigns_display['roas'] = campaigns_display['roas'].apply(format_ratio)
        st.dataframe(campaigns_display, use_container_width=True)
    
    with col2:
        # Campaign efficiency scatter plot
        fig_scatter = px.scatter(
            filtered_campaigns,
            x='spend',
            y='roas',
            size='clicks',
            color='channel',
            hover_data=['campaign', 'tactic', 'state'],
            title=f'Campaign Efficiency (Size = Clicks, Min Spend: {format_currency(min_campaign_spend)})',
            labels={'spend': 'Spend ($)', 'roas': 'ROAS'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Expandable campaign details
    with st.expander("🔍 Detailed Campaign Analysis"):
        st.subheader("Campaign Performance Deep Dive")
        
        # Campaign selection for detailed analysis
        selected_campaign = st.selectbox(
            "Select a campaign for detailed analysis:",
            options=filtered_campaigns['campaign'].tolist(),
            index=0
        )
        
        if selected_campaign:
            campaign_details = filtered_campaigns[filtered_campaigns['campaign'] == selected_campaign].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Channel", campaign_details['channel'])
            with col2:
                st.metric("Tactic", campaign_details['tactic'])
            with col3:
                st.metric("State", campaign_details['state'])
            with col4:
                st.metric("ROAS", format_ratio(campaign_details['roas']))
            
            # Campaign performance over time
            campaign_daily = filtered_marketing[filtered_marketing['campaign'] == selected_campaign].groupby('date').agg({
                'spend': 'sum',
                'attributed_revenue': 'sum',
                'impressions': 'sum',
                'clicks': 'sum'
            }).reset_index()
            
            campaign_daily['roas'] = campaign_daily['attributed_revenue'] / campaign_daily['spend']
            
            fig_campaign_trend = go.Figure()
            fig_campaign_trend.add_trace(go.Scatter(
                x=campaign_daily['date'],
                y=campaign_daily['spend'],
                name='Spend',
                mode='lines+markers',
                line=dict(color='#3b82f6')
            ))
            fig_campaign_trend.add_trace(go.Scatter(
                x=campaign_daily['date'],
                y=campaign_daily['attributed_revenue'],
                name='Attributed Revenue',
                mode='lines+markers',
                line=dict(color='#10b981')
            ))
            fig_campaign_trend.update_layout(
                title=f'{selected_campaign} Performance Over Time',
                height=400
            )
            st.plotly_chart(fig_campaign_trend, use_container_width=True)
    
    # Geographic Performance Analysis
    st.header("🗺️ Geographic Performance Analysis")
    
    # State-wise performance
    state_metrics = filtered_marketing.groupby(['state', 'channel']).agg({
        'spend': 'sum',
        'attributed_revenue': 'sum',
        'impressions': 'sum',
        'clicks': 'sum'
    }).reset_index()
    
    state_metrics['roas'] = state_metrics['attributed_revenue'] / state_metrics['spend']
    state_metrics['ctr'] = state_metrics['clicks'] / state_metrics['impressions']
    state_metrics['cpm'] = state_metrics['spend'] / (state_metrics['impressions'] / 1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # State performance by ROAS
        state_roas = state_metrics.groupby('state')['roas'].mean().sort_values(ascending=False)
        fig_state_roas = px.bar(
            x=state_roas.values,
            y=state_roas.index,
            orientation='h',
            title='Average ROAS by State',
            color=state_roas.values,
            color_continuous_scale='RdYlGn'
        )
        fig_state_roas.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_state_roas, use_container_width=True)
    
    with col2:
        # State spend distribution
        state_spend = state_metrics.groupby('state')['spend'].sum().sort_values(ascending=False)
        fig_state_spend = px.pie(
            values=state_spend.values,
            names=state_spend.index,
            title='Spend Distribution by State'
        )
        fig_state_spend.update_layout(height=400)
        st.plotly_chart(fig_state_spend, use_container_width=True)
    
    # State performance table
    st.subheader("State Performance Summary")
    state_summary = state_metrics.groupby('state').agg({
        'spend': 'sum',
        'attributed_revenue': 'sum',
        'roas': 'mean',
        'ctr': 'mean'
    }).reset_index()
    
    state_summary['spend'] = state_summary['spend'].apply(format_currency)
    state_summary['attributed_revenue'] = state_summary['attributed_revenue'].apply(format_currency)
    state_summary['roas'] = state_summary['roas'].apply(format_ratio)
    state_summary['ctr'] = state_summary['ctr'].apply(format_percentage)
    
    st.dataframe(
        state_summary.sort_values('roas', ascending=False),
        use_container_width=True
    )
    
    # Tactic Performance Analysis
    st.header("🎯 Tactic Performance Analysis")
    
    # Tactic metrics
    tactic_metrics = filtered_marketing.groupby(['tactic', 'channel']).agg({
        'spend': 'sum',
        'attributed_revenue': 'sum',
        'impressions': 'sum',
        'clicks': 'sum'
    }).reset_index()
    
    tactic_metrics['roas'] = tactic_metrics['attributed_revenue'] / tactic_metrics['spend']
    tactic_metrics['ctr'] = tactic_metrics['clicks'] / tactic_metrics['impressions']
    tactic_metrics['cpm'] = tactic_metrics['spend'] / (tactic_metrics['impressions'] / 1000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tactic performance by ROAS
        tactic_roas = tactic_metrics.groupby('tactic')['roas'].mean().sort_values(ascending=False)
        fig_tactic_roas = px.bar(
            x=tactic_roas.values,
            y=tactic_roas.index,
            orientation='h',
            title='Average ROAS by Tactic',
            color=tactic_roas.values,
            color_continuous_scale='RdYlGn'
        )
        fig_tactic_roas.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_tactic_roas, use_container_width=True)
    
    with col2:
        # Tactic performance scatter
        fig_tactic_scatter = px.scatter(
            tactic_metrics,
            x='spend',
            y='roas',
            size='clicks',
            color='channel',
            hover_data=['tactic'],
            title='Tactic Performance (Size = Clicks)',
            labels={'spend': 'Spend ($)', 'roas': 'ROAS'}
        )
        fig_tactic_scatter.update_layout(height=400)
        st.plotly_chart(fig_tactic_scatter, use_container_width=True)
    
    # Time-based Trend Analysis
    st.header("📅 Time-based Trend Analysis")
    
    # Add day of week analysis
    daily_filtered['day_of_week'] = daily_filtered['date'].dt.day_name()
    daily_filtered['week'] = daily_filtered['date'].dt.isocalendar().week
    daily_filtered['month'] = daily_filtered['date'].dt.month_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week performance
        dow_performance = daily_filtered.groupby('day_of_week').agg({
            'spend': 'mean',
            'total_revenue': 'mean',
            'attributed_revenue': 'mean'
        }).reset_index()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_performance['day_of_week'] = pd.Categorical(dow_performance['day_of_week'], categories=day_order, ordered=True)
        dow_performance = dow_performance.sort_values('day_of_week')
        
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(x=dow_performance['day_of_week'], y=dow_performance['spend'], name='Avg Spend', marker_color='#3b82f6'))
        fig_dow.add_trace(go.Bar(x=dow_performance['day_of_week'], y=dow_performance['total_revenue'], name='Avg Revenue', marker_color='#10b981'))
        fig_dow.update_layout(title='Average Performance by Day of Week', height=400)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    with col2:
        # Week over week trend
        weekly_performance = daily_filtered.groupby('week').agg({
            'spend': 'sum',
            'total_revenue': 'sum',
            'attributed_revenue': 'sum'
        }).reset_index()
        
        weekly_performance['roas'] = weekly_performance['attributed_revenue'] / weekly_performance['spend']
        
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(x=weekly_performance['week'], y=weekly_performance['spend'], name='Spend', mode='lines+markers', line=dict(color='#3b82f6')))
        fig_weekly.add_trace(go.Scatter(x=weekly_performance['week'], y=weekly_performance['total_revenue'], name='Revenue', mode='lines+markers', line=dict(color='#10b981')))
        fig_weekly.update_layout(title='Weekly Performance Trend', height=400)
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Profitability Analysis
    st.header("💰 Profitability Analysis")
    
    # Calculate profitability metrics
    daily_filtered['gross_margin'] = daily_filtered['gross_profit'] / daily_filtered['total_revenue']
    daily_filtered['marketing_profit'] = daily_filtered['attributed_revenue'] - daily_filtered['spend']
    daily_filtered['marketing_margin'] = daily_filtered['marketing_profit'] / daily_filtered['attributed_revenue']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gross margin trend
        fig_margin = go.Figure()
        fig_margin.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['gross_margin'], name='Gross Margin', mode='lines+markers', line=dict(color='#10b981')))
        fig_margin.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['marketing_margin'], name='Marketing Margin', mode='lines+markers', line=dict(color='#f59e0b')))
        fig_margin.update_layout(title='Profitability Trends', yaxis_title='Margin', height=400)
        st.plotly_chart(fig_margin, use_container_width=True)
    
    with col2:
        # Profit vs spend scatter
        fig_profit = px.scatter(
            daily_filtered,
            x='spend',
            y='marketing_profit',
            color='gross_margin',
            size='total_revenue',
            hover_data=['date', 'gross_profit'],
            title='Marketing Profit vs Spend (Size = Revenue)',
            labels={'spend': 'Spend ($)', 'marketing_profit': 'Marketing Profit ($)'}
        )
        fig_profit.update_layout(height=400)
        st.plotly_chart(fig_profit, use_container_width=True)
    
    # Customer Cohort Analysis
    st.header("👥 Customer Cohort Analysis")
    
    # Calculate customer metrics
    daily_filtered['new_customer_rate'] = daily_filtered['new_customers'] / daily_filtered['orders']
    daily_filtered['returning_customer_rate'] = 1 - daily_filtered['new_customer_rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer acquisition trend
        fig_customers = go.Figure()
        fig_customers.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['new_customers'], name='New Customers', mode='lines+markers', line=dict(color='#3b82f6')))
        fig_customers.add_trace(go.Scatter(x=daily_filtered['date'], y=daily_filtered['orders'] - daily_filtered['new_orders'], name='Returning Customers', mode='lines+markers', line=dict(color='#10b981')))
        fig_customers.update_layout(title='Customer Acquisition vs Retention', height=400)
        st.plotly_chart(fig_customers, use_container_width=True)
    
    with col2:
        # Customer rate distribution
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Bar(x=daily_filtered['date'], y=daily_filtered['new_customer_rate'], name='New Customer Rate', marker_color='#3b82f6'))
        fig_rates.add_trace(go.Bar(x=daily_filtered['date'], y=daily_filtered['returning_customer_rate'], name='Returning Customer Rate', marker_color='#10b981'))
        fig_rates.update_layout(title='Customer Rate Distribution', yaxis_title='Rate', height=400)
        st.plotly_chart(fig_rates, use_container_width=True)
    
    # Advanced Marketing Metrics
    st.header("📊 Advanced Marketing Metrics")
    
    # Calculate advanced metrics
    daily_filtered['conversion_rate'] = daily_filtered['orders'] / daily_filtered['clicks'] if 'clicks' in daily_filtered.columns else 0
    daily_filtered['rpc'] = daily_filtered['attributed_revenue'] / daily_filtered['clicks'] if 'clicks' in daily_filtered.columns else 0
    daily_filtered['cpa'] = daily_filtered['spend'] / daily_filtered['new_customers']
    daily_filtered['ltv'] = daily_filtered['total_revenue'] / daily_filtered['new_customers']
    
    # Create metrics grid
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        avg_conversion = daily_filtered['conversion_rate'].mean()
        st.metric("Avg Conversion Rate", f"{avg_conversion:.2%}")
    
    with metric_col2:
        avg_rpc = daily_filtered['rpc'].mean()
        st.metric("Avg Revenue per Click", format_currency(avg_rpc))
    
    with metric_col3:
        avg_cpa = daily_filtered['cpa'].mean()
        st.metric("Avg Cost per Acquisition", format_currency(avg_cpa))
    
    with metric_col4:
        avg_ltv = daily_filtered['ltv'].mean()
        st.metric("Avg Customer LTV", format_currency(avg_ltv))
    
    # Performance Benchmarking
    st.header("🎯 Performance Benchmarking")
    
    # Calculate benchmarks
    spend_75th = daily_filtered['spend'].quantile(0.75)
    roas_25th = daily_filtered['roas'].quantile(0.25)
    conversion_25th = daily_filtered['conversion_rate'].quantile(0.25)
    
    # Performance alerts
    st.subheader("Performance Alerts")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        high_spend_days = daily_filtered[daily_filtered['spend'] > spend_75th].shape[0]
        spend_threshold = format_currency(spend_75th)
        st.warning(f"⚠️ {high_spend_days} days with high spend (>75th percentile: {spend_threshold})")
    
    with alert_col2:
        low_roas_days = daily_filtered[daily_filtered['roas'] < roas_25th].shape[0]
        roas_threshold = format_ratio(roas_25th)
        st.error(f"🔴 {low_roas_days} days with low ROAS (<25th percentile: {roas_threshold})")
    
    with alert_col3:
        low_conversion_days = daily_filtered[daily_filtered['conversion_rate'] < conversion_25th].shape[0]
        conversion_threshold = f"{conversion_25th:.2%}"
        st.info(f"ℹ️ {low_conversion_days} days with low conversion (<25th percentile: {conversion_threshold})")
    
    # Interactive Performance Analysis
    st.subheader("Interactive Performance Analysis")
    
    # Performance thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spend_threshold = st.slider(
            "Spend Alert Threshold ($)",
            min_value=float(daily_filtered['spend'].min()),
            max_value=float(daily_filtered['spend'].max()),
            value=float(spend_75th),
            step=1000.0
        )
    
    with col2:
        roas_threshold = st.slider(
            "ROAS Alert Threshold",
            min_value=0.0,
            max_value=float(daily_filtered['roas'].max()),
            value=float(roas_25th),
            step=0.1
        )
    
    with col3:
        conversion_threshold = st.slider(
            "Conversion Rate Alert Threshold (%)",
            min_value=0.0,
            max_value=float(daily_filtered['conversion_rate'].max() * 100),
            value=float(conversion_25th * 100),
            step=0.1
        )
    
    # Dynamic alerts based on user input
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        high_spend_days = daily_filtered[daily_filtered['spend'] > spend_threshold].shape[0]
        st.warning(f"⚠️ {high_spend_days} days with spend > {format_currency(spend_threshold)}")
    
    with alert_col2:
        low_roas_days = daily_filtered[daily_filtered['roas'] < roas_threshold].shape[0]
        st.error(f"🔴 {low_roas_days} days with ROAS < {format_ratio(roas_threshold)}")
    
    with alert_col3:
        low_conversion_days = daily_filtered[daily_filtered['conversion_rate'] < (conversion_threshold/100)].shape[0]
        st.info(f"ℹ️ {low_conversion_days} days with conversion < {conversion_threshold:.1f}%")
    
    # Performance distribution charts
    st.subheader("Performance Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_spend_dist = px.histogram(
            daily_filtered,
            x='spend',
            nbins=20,
            title='Spend Distribution',
            labels={'spend': 'Spend ($)', 'count': 'Number of Days'}
        )
        fig_spend_dist.add_vline(x=spend_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_spend_dist, use_container_width=True)
    
    with col2:
        fig_roas_dist = px.histogram(
            daily_filtered,
            x='roas',
            nbins=20,
            title='ROAS Distribution',
            labels={'roas': 'ROAS', 'count': 'Number of Days'}
        )
        fig_roas_dist.add_vline(x=roas_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_roas_dist, use_container_width=True)
    
    with col3:
        fig_conversion_dist = px.histogram(
            daily_filtered,
            x='conversion_rate',
            nbins=20,
            title='Conversion Rate Distribution',
            labels={'conversion_rate': 'Conversion Rate', 'count': 'Number of Days'}
        )
        fig_conversion_dist.add_vline(x=conversion_threshold/100, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_conversion_dist, use_container_width=True)
    
    # Attribution Analysis
    st.header("🔗 Attribution Analysis")
    
    # Calculate attribution metrics
    total_attributed = daily_filtered['attributed_revenue'].sum()
    total_revenue = daily_filtered['total_revenue'].sum()
    attribution_rate = total_attributed / total_revenue
    
    # Channel attribution
    channel_attribution = filtered_marketing.groupby('channel')['attributed_revenue'].sum().reset_index()
    channel_attribution['attribution_pct'] = channel_attribution['attributed_revenue'] / total_attributed
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attribution funnel
        fig_funnel = go.Figure(go.Funnel(
            y=['Total Revenue', 'Attributed Revenue', 'Marketing Spend'],
            x=[total_revenue, total_attributed, total_spend],
            textinfo="value+percent initial"
        ))
        fig_funnel.update_layout(title='Revenue Attribution Funnel', height=400)
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # Channel attribution
        fig_attr = px.pie(
            channel_attribution,
            values='attribution_pct',
            names='channel',
            title='Channel Attribution Distribution'
        )
        fig_attr.update_layout(height=400)
        st.plotly_chart(fig_attr, use_container_width=True)
    
    # Business Insights
    st.header("💡 Key Business Insights")
    
    # Calculate insights
    attribution_rate = (total_attributed_revenue / total_revenue) * 100 if total_revenue > 0 else 0
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    best_channel = channel_metrics.loc[channel_metrics['roas'].idxmax(), 'channel']
    best_roas = channel_metrics['roas'].max()
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4>📈 Attribution Analysis</h4>
            <p><strong>{attribution_rate:.1f}%</strong> of total revenue is attributed to marketing campaigns</p>
            <p>This indicates {'strong' if attribution_rate > 30 else 'moderate' if attribution_rate > 15 else 'weak'} marketing attribution</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>💰 Revenue Metrics</h4>
            <p>Average Order Value: <strong>{format_currency(avg_order_value)}</strong></p>
            <p>Total Orders: <strong>{total_orders:,}</strong></p>
            <p>New Customers: <strong>{total_new_customers:,}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        st.markdown(f"""
        <div class="insight-box">
            <h4>🏆 Best Performing Channel</h4>
            <p><strong>{best_channel}</strong> leads with ROAS of <strong>{format_ratio(best_roas)}</strong></p>
            <p>Consider increasing budget allocation to this channel</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📊 Marketing Efficiency</h4>
            <p>Overall ROAS: <strong>{format_ratio(overall_roas)}</strong></p>
            <p>Marketing Efficiency Ratio: <strong>{format_ratio(mer)}</strong></p>
            <p>Customer Acquisition Cost: <strong>{format_currency(cac)}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Export functionality
    st.header("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_daily = daily_filtered.to_csv(index=False)
        st.download_button(
            label="Download Daily Combined Data",
            data=csv_daily,
            file_name="daily_combined_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_campaigns = data['campaign_performance'].to_csv(index=False)
        st.download_button(
            label="Download Campaign Performance",
            data=csv_campaigns,
            file_name="campaign_performance.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_channels = channel_metrics.to_csv(index=False)
        st.download_button(
            label="Download Channel Metrics",
            data=csv_channels,
            file_name="channel_metrics.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
