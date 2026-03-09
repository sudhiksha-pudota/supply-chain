"""
WALMART SALES FORECASTING SYSTEM
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import gdown
import pickle

# Page config
st.set_page_config(
    page_title="Walmart Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #0078D7;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0078D7, #00A4EF);
        color: white;
        padding: 1.2rem 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        line-height: 1.2;
    }
    .insight-box {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #0078D7;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #0078D7;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005A9E;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Walmart Sales Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Data-Driven Insights • Model Performance • Business Impact</p>', unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def download_file(file_id, model_path):
    if not os.path.exists(model_path):
        gdown.download(id=file_id, output=model_path, quiet=False,fuzzy=True)

@st.cache_data
def load_data():
    """Load all pre-computed data"""
    
    df_id = '1yJzBrGgxhLRbOmXqsKIBC7BA-IKObQYI'
    df_url = 'https://drive.google.com/file/d/1yJzBrGgxhLRbOmXqsKIBC7BA-IKObQYI'
    df_path = 'walmart_final_with_trends.csv'
    
    results_id = '1NqfK44tLNt6DrMv6ejoDpptDuuFOxwJu'
    results_url = 'https://drive.google.com/file/d/1NqfK44tLNt6DrMv6ejoDpptDuuFOxwJu'
    results_path = 'strategic_model_results.csv'

    analysis_id = '1U5mbosl9YqrVwWp_8XdynCHSQDgTnz8Z'
    analysis_url = 'https://drive.google.com/file/d/1U5mbosl9YqrVwWp_8XdynCHSQDgTnz8Z'
    analysis_path = 'department_analysis.csv'

    data_id = '1SXAjgcr47v7zoVX84d2wkUrDWyAHrxdH'
    data_url = 'https://drive.google.com/file/d/1SXAjgcr47v7zoVX84d2wkUrDWyAHrxdH'
    data_path = 'walmart_final_addnl.csv'

    features_id = '10ErLm1fnW7w9PYFwCTKHaRagrMdwZWQg'
    features_url = 'https://drive.google.com/file/d/10ErLm1fnW7w9PYFwCTKHaRagrMdwZWQg'
    features_path = 'feature_importance_xgb.csv'

    uplift_id = '1VVsf9ZYb1ewFkUOrRrJmPHZlcltLd_m2'
    uplift_url = 'https://drive.google.com/file/d/1VVsf9ZYb1ewFkUOrRrJmPHZlcltLd_m2'
    uplift_path = 'store_dept_uplift_data.csv'
    
    try:
        # Main sales data
        download_file(df_id, df_path)
        df = pd.read_csv(df_path)
        
        # Model results
        download_file(results_id, results_path)
        results = pd.read_csv(results_path)
        
        # Department analysis
        download_file(analysis_id, analysis_path)
        analysis = pd.read_csv(analysis_path)
        
        # Additional features
        download_file(data_id, data_path)
        data = pd.read_csv(data_path)
        
        # Feature importance
        try:
            download_file(features_id, features_path)
            features = pd.read_csv(features_path)
        except:
            features = None

        # Uplift data
        download_file(uplift_id, uplift_path)
        store_dept_uplift = pd.read_csv(uplift_path)
        
        return df, results, analysis, data, features, store_dept_uplift
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

df, results, analysis, data, features, store_dept_uplift = load_data()

if df is None:
    st.error("Data files not found. Please check file paths.")
    st.stop()

# ============================================
# HELPER FUNCTIONS 
# ============================================
@st.cache_data
def calculate_department_stats():
    """Calculate real department statistics"""
    stats = df.groupby(['Store', 'Dept']).agg({
        'Weekly_Sales': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
    stats.columns = ['Mean', 'Std', 'Min', 'Max', 'Weeks']
    stats = stats.reset_index()
    
    # Calculate Z-scores to find statistical outliers
    stats['Z_Score'] = (stats['Mean'] - stats['Mean'].mean()) / stats['Mean'].std()
    stats['Is_Outlier'] = abs(stats['Z_Score']) > 2
    
    # Add volatility from analysis
    stats = stats.merge(analysis[['Store', 'Dept', 'CV', 'HolidayLift%', 'ZeroPct']], 
                        on=['Store', 'Dept'], how='left')
    
    return stats

@st.cache_data
def calculate_inventory_impact(profit_margin=0.25, holding_cost=0.25):
    """Calculate real inventory impact based on forecast errors"""
    # Get naive forecast results
    naive_results = results[results['Model'] == 'Naive'].copy()
    
    # Get average sales per department
    dept_sales = df.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
    dept_sales.columns = ['Store', 'Dept', 'Avg_Weekly_Sales']
    
    # Merge
    impact = naive_results.merge(dept_sales, on=['Store', 'Dept'])
    
    # Calculate annual error in dollars
    impact['Annual_Sales'] = impact['Avg_Weekly_Sales'] * 52
    impact['Error_Dollars'] = (impact['MAPE'] / 100) * impact['Annual_Sales']
    
    # Add volatility from analysis
    impact = impact.merge(analysis[['Store', 'Dept', 'CV']], on=['Store', 'Dept'], how='left')
    impact['CV'].fillna(0.5, inplace=True)
    
    # More volatile departments have more stockouts (30-70% range)
    impact['Under_Forecast_Pct'] = 0.3 + (impact['CV'] * 0.4)
    impact['Under_Forecast_Pct'] = impact['Under_Forecast_Pct'].clip(0.3, 0.7)
    
    # Calculate costs
    impact['Stockout_Cost'] = (impact['Error_Dollars'] * 
                               impact['Under_Forecast_Pct'] * 
                               profit_margin * 2)  # Stockout costs more
    
    impact['Holding_Cost'] = (impact['Error_Dollars'] * 
                              (1 - impact['Under_Forecast_Pct']) * 
                              holding_cost)
    
    impact['Total_Impact'] = impact['Stockout_Cost'] + impact['Holding_Cost']
    
    return impact

@st.cache_data
def calculate_improvement_potential():
    """Calculate potential savings from using best models"""
    # Best model for each department
    best_models = results.loc[results.groupby(['Store', 'Dept'])['MAPE'].idxmin()]
    
    # Current naive performance
    naive = results[results['Model'] == 'Naive'].copy()
    
    # Merge
    potential = naive.merge(
        best_models[['Store', 'Dept', 'MAPE']], 
        on=['Store', 'Dept'], 
        suffixes=('_Naive', '_Best')
    )
    
    # Add sales data
    dept_sales = df.groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
    dept_sales.columns = ['Store', 'Dept', 'Avg_Weekly_Sales']
    potential = potential.merge(dept_sales, on=['Store', 'Dept'])
    
    # Calculate potential savings
    potential['Annual_Sales'] = potential['Avg_Weekly_Sales'] * 52
    potential['Current_Error'] = (potential['MAPE_Naive'] / 100) * potential['Annual_Sales']
    potential['Best_Error'] = (potential['MAPE_Best'] / 100) * potential['Annual_Sales']
    potential['Potential_Savings'] = potential['Current_Error'] - potential['Best_Error']
    
    return potential

# ============================================
# SIDEBAR - Navigation
# ============================================

image_id = '1VlUsL1ur5xQrH4agOCPbSUdk7sUJJ9Fh'
image_url = 'https://drive.google.com/file/d/1VlUsL1ur5xQrH4agOCPbSUdk7sUJJ9Fh'
image_path = 'walmart_logo.png'

try:
    from PIL import Image
    download_file(image_id, image_path)
    logo = Image.open(image_path)
    st.sidebar.image(logo, width=200)
except:
    st.sidebar.markdown("# Walmart")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Performance Dashboard",
        "Department Analyzer",
        "Forecast Predictor",
        "Outlier Detection",
        "Inventory Impact",
        "Business Value",
        "Performance Reports",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success(f"{len(analysis)} Departments")
st.sidebar.info(f"{results['Model'].nunique()} Models")
st.sidebar.info(f"{len(df)} Weekly Records")

# Calculate real-time stats
dept_stats = calculate_department_stats()
inventory_impact = calculate_inventory_impact()
improvement_potential = calculate_improvement_potential()

# ============================================
# PAGE 1: PERFORMANCE DASHBOARD
# ============================================
if page == "Performance Dashboard":

    df['Date'] = pd.to_datetime(df['Date'], format = "%d/%m/%Y")
    
    # Extract date features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Date range filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    filtered_df = df.copy()
    
    tab1, tab2, tab3 = st.tabs(["Sales Analysis", "Time Series Analysis","Model Performance"])

    with tab1:
        st.header("Sales Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_sales = filtered_df['Weekly_Sales'].sum()
            st.metric("Total Weekly Sales", f"${total_sales:,.2f}")
        
        with col2:
            avg_sales = filtered_df['Weekly_Sales'].mean()
            st.metric("Average Weekly Sales", f"${avg_sales:,.2f}")
        
        with col3:
            median_sales = filtered_df['Weekly_Sales'].median()
            st.metric("Median Weekly Sales", f"${median_sales:,.2f}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribution of Weekly Sales")
            fig = px.histogram(
                filtered_df, 
                x='Weekly_Sales', 
                nbins=50,
                title="Sales Distribution",
                labels={'Weekly_Sales': 'Weekly Sales ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sales by Holiday Period")
            holiday_sales = filtered_df.groupby('IsHoliday')['Weekly_Sales'].agg(['mean', 'sum']).reset_index()
            holiday_sales['IsHoliday'] = holiday_sales['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Average Sales', 'Total Sales'))
            
            fig.add_trace(
                go.Bar(x=holiday_sales['IsHoliday'], y=holiday_sales['mean'], name='Avg Sales'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=holiday_sales['IsHoliday'], y=holiday_sales['sum'], name='Total Sales'),
                row=1, col=2
            )
            
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.header("Store Analysis")

        # Store performance
        store_performance = filtered_df.groupby('Store').agg({
            'Weekly_Sales': ['sum', 'mean', 'count'],
            'Type': 'first',
            'Size': 'first'
        }).round(2)
        
        store_performance.columns = ['Total_Sales', 'Avg_Sales', 'Num_Records', 'Type', 'Size']
        store_performance = store_performance.reset_index()
        store_performance = store_performance.sort_values('Total_Sales', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Store Performance by Type")
            fig = px.box(
                filtered_df, 
                x='Type', 
                y='Weekly_Sales',
                title="Sales Distribution by Store Type",
                labels={'Weekly_Sales': 'Weekly Sales ($)', 'Type': 'Store Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Store Size Distribution by Type")
            fig = px.box(
                filtered_df.drop_duplicates('Store'),
                x='Type', 
                y='Size',
                title="Store Size Distribution by Type",
                labels={'Size': 'Store Size (sq ft)', 'Type': 'Store Type'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Time Series Analysis")
        
        # Aggregate sales by date
        daily_sales = filtered_df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        
        st.subheader("Sales Over Time")
        
        # Resampling options
        resample_period = st.radio(
            "Resample Period",
            options=['Daily', 'Weekly', 'Monthly']
        )
        
        if resample_period == 'Weekly':
            daily_sales['Period'] = daily_sales['Date'].dt.to_period('W').astype(str)
            ts_data = daily_sales.groupby('Period')['Weekly_Sales'].sum().reset_index()
        elif resample_period == 'Monthly':
            daily_sales['Period'] = daily_sales['Date'].dt.to_period('M').astype(str)
            ts_data = daily_sales.groupby('Period')['Weekly_Sales'].sum().reset_index()
        else:
            ts_data = daily_sales
            ts_data['Period'] = ts_data['Date'].astype(str)
        
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        fig = px.line(
            ts_data,
            x='Period',
            y='Weekly_Sales',
            title=f"{resample_period} Sales Trend",
            labels={'Weekly_Sales': 'Total Sales ($)', 'Period': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly and yearly patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Monthly Sales Pattern")
            monthly_avg = filtered_df.groupby('Month')['Weekly_Sales'].mean().reset_index()
            fig = px.bar(
                monthly_avg,
                x='Month',
                y='Weekly_Sales',
                title="Average Sales by Month",
                labels={'Weekly_Sales': 'Average Sales ($)', 'Month': 'Month'}
            )
            fig.update_xaxes(
                    tickmode = 'array',
                    tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    ticktext = month_labels
                )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Yearly Sales Comparison")
            yearly_sales = filtered_df.groupby('Year')['Weekly_Sales'].sum().reset_index()
            fig = px.bar(
                yearly_sales,
                x='Year',
                y='Weekly_Sales',
                title="Total Sales by Year",
                labels={'Weekly_Sales': 'Total Sales ($)', 'Year': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Model Performance Dashboard")
        
        # Key metrics
        naive_results = results[results['Model'] == 'Naive']
        avg_mape = naive_results['MAPE'].mean()
        best_models = results.loc[results.groupby(['Store', 'Dept'])['MAPE'].idxmin()]
        best_avg_mape = best_models['MAPE'].mean()
        
        col1, col2, col3, col4 = st.columns(4)

        # Calculate Naive wins (within 5% of best)
        best_mape = results.groupby(['Store', 'Dept'])['MAPE'].transform('min')
        naive_wins = len(results[(results['Model'] == 'Naive') & (results['MAPE'] <= best_mape * 1.05)])
        
        best_dept = naive_results.loc[naive_results['MAPE'].idxmin()]
        worst_dept = naive_results.loc[naive_results['MAPE'].idxmax()]
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Average MAPE</div>
                <div class="metric-value">{avg_mape:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Best Dept</div>
                <div class="metric-value">Store {int(best_dept['Store'])} / {int(best_dept['Dept'])}</div>
                <div class="metric-delta">{best_dept['MAPE']:.1f}% MAPE</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Worst Dept</div>
                <div class="metric-value">Store {int(worst_dept['Store'])} / {int(worst_dept['Dept'])}</div>
                <div class="metric-delta">{worst_dept['MAPE']:.1f}% MAPE</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            improvement = avg_mape - best_avg_mape
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Potential Improvement</div>
                <div class="metric-value">{improvement:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Key insight
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        best_model_counts = best_models['Model'].value_counts()
        top_model = best_model_counts.index[0]
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance Comparison")
            model_avg = results.groupby('Model')['MAPE'].mean().sort_values().reset_index()
            fig = px.bar(model_avg, x='MAPE', y='Model', orientation='h',
                        color='MAPE', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Which Model Wins Most?")
            win_counts = best_models['Model'].value_counts().reset_index()
            win_counts.columns = ['Model', 'Count']
            fig = px.pie(win_counts, values='Count', names='Model')
            st.plotly_chart(fig, use_container_width=True)
        
        # MAPE Distribution
        st.subheader("MAPE Distribution Across Departments")
        fig = px.histogram(naive_results, x='MAPE', nbins=30,
                        title="Naive Forecast MAPE Distribution")
        fig.add_vline(x=avg_mape, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {avg_mape:.1f}%")
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 2: DEPARTMENT ANALYZER
# ============================================
elif page == "Department Analyzer":
    st.header("Department Performance Analyzer")
    
    tab1, tab2 = st.tabs(["Performance Analysis", "Feature Importance"])

    with tab1:

        col1, col2 = st.columns([2,3])
    
        with col1:
            st.subheader("Select Department")
            column1, column2 = st.columns(2)
            with column1:
                store = st.selectbox("Store", sorted(data['Store'].unique()), key = "perf_store")
            with column2:
                dept_options = sorted(data[data['Store'] == store]['Dept'].unique())
                dept = st.selectbox("Department", dept_options, key = "perf_dept")
            
            # Get real department info
            dept_info = data[(data['Store'] == store) & (data['Dept'] == dept)].iloc[0]
            dept_stat = dept_stats[(dept_stats['Store'] == store) & (dept_stats['Dept'] == dept)].iloc[0]
            uplift_info = store_dept_uplift[(store_dept_uplift['Store'] == store) & (store_dept_uplift['Dept'] == dept)]
            holiday_treatment = uplift_info['Holiday_Uplift']

            st.markdown("---")
            st.subheader("Characteristics")

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Holiday Lift %", f"{dept_info['HolidayLift%']:.1f}%")
                st.metric("Volatility (CV)", f"{dept_info['CV']:.2f}")
            with c2:
                st.metric("Zero Sales Weeks", f"{dept_info['ZeroPct']:.1f}%")
                st.metric("Average Weekly Sales", f"${dept_stat['Mean']:,.0f}")
            
            st.metric("Holiday Lift (Sales)", f"${holiday_treatment.iloc[0]:,.0f}")

            # Show if outlier
            if dept_stat['Is_Outlier']:
                st.warning("This department is a statistical outlier")
            
            dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
            dept_data['Date'] = pd.to_datetime(dept_data['Date'], format = "%d/%m/%Y")

            if not dept_data.empty:
                dept_data = dept_data.sort_values('Date')
                
                # Model performance
                dept_perf = results[(results['Store'] == store) & (results['Dept'] == dept)]
                if not dept_perf.empty:
                    st.subheader("Model Performance")
                    fig = px.bar(dept_perf, x='Model', y='MAPE',
                                color='MAPE', color_continuous_scale='RdYlGn_r')
                    st.plotly_chart(fig, use_container_width=True)

            # Recommendation
            st.markdown("---")
            st.subheader("Recommendation")
            
            if dept_info['Recommended'] == 'XGBOOST':
                st.success(f"**{dept_info['Recommended']}** (Confidence: {dept_info['Confidence']:.0%})")
                st.info("Stable department - ML can help")
            elif dept_info['Recommended'] == 'NAIVE':
                st.warning(f"**{dept_info['Recommended']}** (Confidence: {dept_info['Confidence']:.0%})")
                st.info("Holiday-sensitive - Keep it simple")
            else:
                st.info(f"**{dept_info['Recommended']}** (Confidence: {dept_info['Confidence']:.0%})")
                st.info("Borderline case - use ensemble")
        
        with col2:
            # Show actual sales data
            dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
            dept_data['Date'] = pd.to_datetime(dept_data['Date'], format = "%d/%m/%Y")

            if not dept_data.empty:
                dept_data = dept_data.sort_values('Date')
                
                # Time series
                fig = px.line(dept_data, x='Date', y='Weekly_Sales',
                            title=f"Store {store}, Dept {dept} Sales History")
                
                # Highlight holidays
                holidays = dept_data[dept_data['IsHoliday'] == 1]
                if not holidays.empty:
                    fig.add_scatter(x=holidays['Date'], y=holidays['Weekly_Sales'],
                                mode='markers', name='Holidays',
                                marker=dict(color='red', size=8))
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly pattern
                dept_data['Month'] = dept_data['Date'].dt.month
                monthly = dept_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
                    
                month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                fig = px.line(
                    monthly,
                    x='Month',
                    y='Weekly_Sales',
                    title="Average Monthly Pattern",
                )
                fig.add_vrect(x0=11, x1=12, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Holiday Season")
                fig.update_xaxes(
                    tickmode = 'array',
                    tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    ticktext = month_labels
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if features is None:
            st.warning("Feature importance data not available for this dataset")
            st.stop()
        
        st.header("Feature Importance Analysis")
        st.info("Based on XGBoost model for departments where it performs best")
        
        # Department selection
        col1, col2 = st.columns(2)
        
        with col1:
            store = st.selectbox("Store", sorted(features['Store'].unique()))
        
        with col2:
            dept_options = sorted(features[features['Store'] == store]['Dept'].unique())
            dept = st.selectbox("Department", dept_options)
        
        # Get features for this department
        dept_features = features[(features['Store'] == store) & 
                                (features['Dept'] == dept)].copy()
        
        if not dept_features.empty:
            # Sort by importance
            dept_features = dept_features.sort_values('Importance', ascending=False)
            
            # Show top features
            st.subheader(f"Top Features for Store {store}, Dept {dept}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Bar chart of top features
                top_n = st.slider("Number of features to show", 1, 10, 5)
                top_features = dept_features.head(top_n)
                
                fig = px.bar(top_features, x='Importance', y='Feature',
                            orientation='h', title=f"Top {top_n} Features",
                            color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Summary")
                st.metric("Total Features", len(dept_features))
                st.metric("Top Feature", top_features.iloc[0]['Feature'])
                st.metric("Top Importance", f"{top_features.iloc[0]['Importance']:.3f}")
                
                # Cumulative importance
                cumulative = top_features['Importance'].sum() / dept_features['Importance'].sum() * 100
                st.metric(f"Top {top_n} Features Account For", f"{cumulative:.1f}%")
            
            # Full table
            with st.expander("View All Features"):
                st.dataframe(dept_features, use_container_width=True)
            
            # Interpretation
            st.markdown("### Interpretation")
            st.markdown(f"""
            - **Top feature:** {top_features.iloc[0]['Feature']} (importance: {top_features.iloc[0]['Importance']:.3f})
            - This means {top_features.iloc[0]['Feature']} is the strongest predictor of sales for this department
            - The top {top_n} features explain {cumulative:.1f}% of the model's predictive power
            """)
        else:
            st.warning("No feature importance data for this department")


elif page == "Forecast Predictor":
    st.header("Real-Time Sales Predictor")

    st.markdown("Enter values to get weekly forecast")
        
    col1, col2 = st.columns([1, 1])
        
    with col1:
        st.subheader("Input Parameters")
            
        store = st.selectbox("Store Number", options=sorted(df['Store'].unique()), key='pred_store')
        dept_options = sorted(df[df['Store'] == store]['Dept'].unique())
        dept = st.selectbox("Department Number", options=dept_options, key='pred_dept')
            
        # Get department info
        dept_info = data[(data['Store'] == store) & (data['Dept'] == dept)]

        if (dept_info['Recommended'] == 'XGBOOST').any() or (dept_info['Recommended'] == 'ENSEMBLE').any():
            temperature = st.number_input('Temperature', format = "%.2f")
            fuel_price = st.number_input('Fuel Price', format = "%.3f")
            cpi = st.number_input('Consumer Price Index', format = "%.4f")
            unemployment = st.number_input('Unemployment Rate', format = "%.3f")
            size = st.number_input('Size', step = 1)
            is_holiday = st.checkbox("Next week is a holiday?", key = 'pred_xgboost_hold')

            model_id = '1SbPkdUhLYzUJxvl7XszGVGFAReKjeQhN'
            model_url = 'https://drive.google.com/file/d/1SbPkdUhLYzUJxvl7XszGVGFAReKjeQhN'
            model_path = 'best_xgboost_model.pkl'

            download_file(model_id, model_path)

            with open(model_path, 'rb') as file:
                xgboost_model = pickle.load(file)

            input_data = pd.DataFrame({
                'Store': [store],
                'Dept': [dept],
                'Temperature': [temperature],
                'Fuel_Price': [fuel_price],
                'CPI': [cpi],
                'Unemployment': [unemployment],
                'IsHoliday': [is_holiday],
                'Size': [size]
            })

            prediction = xgboost_model.predict(input_data)
            val = prediction[0]
            
        else: # NAIVE
            val = st.number_input("Last Week's Sales ($)", min_value=0, value=10000, step=1000, key = 'pred_sales')
            is_holiday = st.checkbox("Next week is a holiday?", key = 'pred_hol')
            
        if st.button("Generate Forecast", type="primary", use_container_width=True, key = 'trial_button'):
                
            if len(dept_info) == 0:
                st.error("Department not found in analysis")
            else:
                info = dept_info.iloc[0]
                    
                # Calculate forecast
                if is_holiday and info['HolidayLift%'] > 30:
                    predicted = val * (1 + info['HolidayLift%'] / 100)
                    method = f"Holiday Adjusted (+{info['HolidayLift%']:.1f}%)"
                else:
                    predicted = val
                    if (dept_info['Recommended'] == 'XGBOOST').any() or (dept_info['Recommended'] == 'ENSEMBLE').any():
                        method = "XGBoost"
                    else:
                        method = "Naive Forecast (last week)"
                    
                # Store in session state
                st.session_state['predicted'] = predicted
                st.session_state['method'] = method
                st.session_state['info'] = info
        
    with col2:
        st.subheader("Forecast Result")
            
        if 'predicted' in st.session_state:
            # Main prediction display
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0078D7, #00A4EF); padding: 2rem; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 3rem;">${st.session_state['predicted']:,.0f}</h1>
                <p style="color: white; opacity: 0.9;">Predicted Sales</p>
            </div>
            """, unsafe_allow_html=True)
                
            # Method
            st.info(f"Method: {st.session_state['method']}")
                
            # Confidence
            info = st.session_state['info']
            if info['CV'] < 0.3:
                st.success("High Confidence - Stable Department")
            elif info['CV'] > 0.8:
                st.warning("Low Confidence - Highly Volatile")
            else:
                st.info("Medium Confidence")
                
            # Holiday alert
            if is_holiday and info['HolidayLift%'] > 30:
                st.warning(f"Holiday Alert: This department typically sees +{info['HolidayLift%']:.1f}% sales during holidays")
        else:
            # Placeholder
            st.markdown("""
            <div style="background: #f0f2f6; padding: 3rem; border-radius: 15px; text-align: center; color: #999;">
                <h2>---</h2>
                <p>Enter parameters and click Generate</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Historical context
    if 'predicted' in st.session_state:
        st.markdown("---")
        st.subheader("Historical Context")
            
        # Get historical data
        hist_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
        if len(hist_data) > 0:
            hist_data['Date'] = pd.to_datetime(hist_data['Date'], format = "%d/%m/%Y")
                
            # Create forecast point
            next_date = hist_data['Date'].max() + timedelta(days=7)
            forecast_df = pd.DataFrame({
                'Date': [next_date],
                'Weekly_Sales': [st.session_state['predicted']],
                'Type': ['Forecast']
            })
            hist_data['Type'] = 'Historical'
                
            # Combine
            plot_data = pd.concat([hist_data[['Date', 'Weekly_Sales', 'Type']], forecast_df])
                
            fig = px.line(
                plot_data, 
                x='Date', 
                y='Weekly_Sales',
                color='Type',
                title="Recent History + Forecast",
                color_discrete_map={'Historical': '#0078D7', 'Forecast': '#FF4444'}
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE 3: OUTLIER DETECTION
# ============================================
elif page == "Outlier Detection":
    st.header("Statistical Outlier Detection")
    st.markdown("Finding departments with unusual sales patterns")
    
    # Show outlier summary
    outlier_depts = dept_stats[dept_stats['Is_Outlier'] == True]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Departments", len(dept_stats))
    with col2:
        st.metric("Outliers Detected", len(outlier_depts))
    with col3:
        st.metric("Percentage", f"{len(outlier_depts)/len(dept_stats)*100:.1f}%")
    
    tab1, tab2, tab3 = st.tabs(["Explore Outliers", "Compare Patterns", "Business Impact"])
    
    with tab1:
        st.subheader("Departments with Unusual Sales Patterns")
        
        if not outlier_depts.empty:
            # Sort by how extreme
            outlier_depts_sorted = outlier_depts.sort_values(['Store', 'Dept'], ascending=True)
            
            # Let user select
            selected = st.selectbox(
                "Select an outlier department to analyze",
                [f"Store {row['Store']}, Dept {row['Dept']} (Mean: ${row['Mean']:,.0f}, Z={row['Z_Score']:.1f})" 
                 for _, row in outlier_depts_sorted.iterrows()]
            )
            
            # Parse selection
            store = int(selected.split('Store ')[1].split(',')[0])
            dept = int(selected.split('Dept ')[1].split(' ')[0])
            
            dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
                
            fig = px.line(dept_data, x='Date', y='Weekly_Sales',
                          title=f"Store {store}, Dept {dept} - Sales Pattern")
                
            
            stats = dept_stats[(dept_stats['Store'] == store) & 
                                  (dept_stats['Dept'] == dept)].iloc[0]
                
            st.markdown("### Department Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Average Sales", f"${stats['Mean']:,.0f}")

            with col2:
                st.metric("Standard Deviation", f"${stats['Std']:,.0f}")
            
            with col3:
                st.metric("Max Sales", f"${stats['Max']:,.0f}")
            
            with col4:
                st.metric("Min Sales", f"${stats['Min']:,.0f}")
            
            with col5:
                st.metric("Z-Score", f"{stats['Z_Score']:.2f}")
            
            # Highlight weekly outliers within this department
            q1 = dept_data['Weekly_Sales'].quantile(0.25)
            q3 = dept_data['Weekly_Sales'].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 1.5 * iqr
                
            weekly_outliers = dept_data[dept_data['Weekly_Sales'] > upper]
            if not weekly_outliers.empty:
                fig.add_scatter(x=weekly_outliers['Date'], y=weekly_outliers['Weekly_Sales'],
                                mode='markers', name='Weekly Outliers',
                                marker=dict(color='red', size=10))
                
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)

            with col1:
                dept_data = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
                dept_data['Date'] = pd.to_datetime(dept_data['Date'], format = "%d/%m/%Y")

                if len(dept_data) > 0:
                    # Monthly pattern
                    dept_data['Month'] = dept_data['Date'].dt.month
                    monthly = dept_data.groupby('Month')['Weekly_Sales'].mean().reset_index()
                        
                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                    fig = px.line(
                        monthly,
                        x='Month',
                        y='Weekly_Sales',
                        title="Average Monthly Pattern",
                    )
                    fig.add_vrect(x0=11, x1=12, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Holiday Season")
                    fig.update_xaxes(
                        tickmode = 'array',
                        tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                        ticktext = month_labels
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No historical data available for this department")
            
            with col2:
                # Box plot comparison
                non_holiday_data = dept_data[dept_data['IsHoliday'] == 0]['Weekly_Sales']
                holiday_data = dept_data[dept_data['IsHoliday'] == 1]['Weekly_Sales']       

                comp_df = pd.DataFrame({
                    'Sales': pd.concat([non_holiday_data, holiday_data]),
                    'Holidays': ['Non-Holiday'] * len(non_holiday_data) + ['Holiday'] * len(holiday_data)
                })
                
                fig = px.box(comp_df, x='Holidays', y='Sales', color='Holidays',
                            title="Holiday vs Non-Holiday Sales")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No statistical outliers found in your data")
    
    with tab2:
        st.subheader("Outlier vs Normal Department Comparison")
        
        if not outlier_depts.empty:
            # Pick a random normal department for comparison
            normal_depts = dept_stats[dept_stats['Is_Outlier'] == False]
            
            if not normal_depts.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    outlier_sample = outlier_depts.sample(1).iloc[0]
                    st.markdown(f"**Outlier:** Store {outlier_sample['Store']}, Dept {outlier_sample['Dept']}")
                    st.metric("Mean Sales", f"${outlier_sample['Mean']:,.0f}")
                    st.metric("Std Dev", f"${outlier_sample['Std']:,.0f}")
                
                with col2:
                    normal_sample = normal_depts.sample(1).iloc[0]
                    st.markdown(f"**Normal:** Store {normal_sample['Store']}, Dept {normal_sample['Dept']}")
                    st.metric("Mean Sales", f"${normal_sample['Mean']:,.0f}")
                    st.metric("Std Dev", f"${normal_sample['Std']:,.0f}")
                
                # Box plot comparison
                outlier_data = df[(df['Store'] == outlier_sample['Store']) & 
                                 (df['Dept'] == outlier_sample['Dept'])]['Weekly_Sales']
                normal_data = df[(df['Store'] == normal_sample['Store']) & 
                                (df['Dept'] == normal_sample['Dept'])]['Weekly_Sales']
                
                comp_df = pd.DataFrame({
                    'Sales': pd.concat([outlier_data, normal_data]),
                    'Type': ['Outlier'] * len(outlier_data) + ['Normal'] * len(normal_data)
                })
                
                fig = px.box(comp_df, x='Type', y='Sales', color='Type',
                            title="Sales Distribution Comparison")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Business Impact of Outliers")
        
        # Calculate impact
        outlier_impact = inventory_impact[inventory_impact['Store'].isin(outlier_depts['Store']) &
                                         inventory_impact['Dept'].isin(outlier_depts['Dept'])]
        
        normal_impact = inventory_impact[~inventory_impact['Store'].isin(outlier_depts['Store']) |
                                        ~inventory_impact['Dept'].isin(outlier_depts['Dept'])]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Error Cost - Outliers", 
                     f"${outlier_impact['Total_Impact'].mean():,.0f}")
        with col2:
            st.metric("Avg Error Cost - Normal", 
                     f"${normal_impact['Total_Impact'].mean():,.0f}",
                     delta=f"${outlier_impact['Total_Impact'].mean() - normal_impact['Total_Impact'].mean():,.0f}")
        
        st.info("Outlier departments have significantly higher forecast error costs and need specialized attention")

# ============================================
# PAGE 4: INVENTORY IMPACT
# ============================================
elif page == "Inventory Impact":
    st.header("Forecast Error Impact on Inventory")
    st.markdown("Based on actual model performance")
    
    # User-adjustable parameters
    with st.sidebar.expander("Inventory Parameters", expanded=False):
        profit_margin = st.slider("Profit Margin %", 10, 40, 25) / 100
        holding_cost = st.slider("Holding Cost %", 15, 35, 25) / 100
        stockout_penalty = st.slider("Stockout Penalty (x lost profit)", 1.5, 3.0, 2.0, 0.1)
    
    # Recalculate with user parameters
    impact = calculate_inventory_impact(profit_margin, holding_cost)
    impact['Stockout_Cost'] = impact['Error_Dollars'] * impact['Under_Forecast_Pct'] * profit_margin * stockout_penalty
    impact['Total_Impact'] = impact['Stockout_Cost'] + impact['Holding_Cost']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Annual Impact", f"${impact['Total_Impact'].sum():,.0f}")
    with col2:
        st.metric("Stockout Costs", f"${impact['Stockout_Cost'].sum():,.0f}")
    with col3:
        st.metric("Holding Costs", f"${impact['Holding_Cost'].sum():,.0f}")
    with col4:
        st.metric("Total Error Dollars", f"${impact['Error_Dollars'].sum():,.0f}")
    
    # Visualize
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 departments by impact
        st.subheader("Top 10 Departments by Inventory Impact")
        top_impact = impact.nlargest(10, 'Total_Impact')[['Store', 'Dept', 'MAPE', 'Total_Impact']].copy()
        top_impact['Total_Impact'] = top_impact['Total_Impact'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(top_impact, use_container_width=True)
    
    with col2:
        # Impact distribution
        fig = px.histogram(impact, x='Total_Impact', nbins=30,
                          title="Distribution of Inventory Impact")
        st.plotly_chart(fig, use_container_width=True)
    
    # Breakdown by volatility
    st.subheader("Impact by Department Volatility")
    
    impact['Volatility_Bin'] = pd.cut(impact['CV'], 
                                      bins=[0, 0.3, 0.6, 1.0, float('inf')],
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    
    vol_impact = impact.groupby('Volatility_Bin').agg({
        'Total_Impact': 'sum',
        'Store': 'count'
    }).reset_index()
    vol_impact.columns = ['Volatility', 'Total_Impact', 'Dept_Count']
    
    fig = px.bar(vol_impact, x='Volatility', y='Total_Impact',
                color='Dept_Count', title="Inventory Impact by Volatility Level",
                labels={'Total_Impact': 'Total Impact ($)'})
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE 5: BUSINESS VALUE
# ============================================
elif page == "Business Value":
    st.header("Business Value of Improved Forecasting")
    
    # Calculate real metrics
    total_sales = df['Weekly_Sales'].sum() * 52
    naive_mape = results[results['Model'] == 'Naive']['MAPE'].mean()
    best_mape = results.groupby(['Store', 'Dept'])['MAPE'].min().mean()
    
    current_error = total_sales * (naive_mape / 100)
    best_error = total_sales * (best_mape / 100)
    savings = current_error - best_error
    
    tab1, tab2 = st.tabs(["Current Impact", "Improvement Potential"])
    
    with tab1:
        st.subheader("Current State - Naive Forecast")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Sales", f"${total_sales:,.0f}")
        with col2:
            st.metric("Avg Forecast Error", f"{naive_mape:.1f}%")
        with col3:
            st.metric("Error in Dollars", f"${current_error:,.0f}")
        
        # Error distribution
        fig = px.histogram(results[results['Model'] == 'Naive'], x='MAPE', nbins=30,
                          title="Distribution of Forecast Errors")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        st.subheader("Estimated Annual Costs")
        
        # Simple cost model (50% over, 50% under)
        under_cost = current_error * 0.5 * 0.25 * 2  # Lost profit * penalty
        over_cost = current_error * 0.5 * 0.25       # Holding cost
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stockout Costs", f"${under_cost:,.0f}")
            st.caption("Lost sales + customer goodwill")
        with col2:
            st.metric("Holding Costs", f"${over_cost:,.0f}")
            st.caption("Excess inventory carrying costs")
    
    with tab2:
        st.subheader("Potential with Best Model Selection")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Possible MAPE", f"{best_mape:.1f}%",
                     delta=f"{- (naive_mape - best_mape):.1f}%")
        with col2:
            st.metric("Best Error Cost", f"${best_error:,.0f}")
        with col3:
            st.metric("Potential Savings", f"${savings:,.0f}",
                     delta=f"{(savings/current_error*100):.1f}%")
        
        # Show which models would help
        best_models = results.loc[results.groupby(['Store', 'Dept'])['MAPE'].idxmin()]
        model_counts = best_models['Model'].value_counts().reset_index()
        model_counts.columns = ['Model', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(model_counts, values='Count', names='Model',
                        title="Optimal Model Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Model Impact")
            for _, row in model_counts.iterrows():
                st.metric(row['Model'], f"{row['Count']} departments")

# ============================================
# PAGE 6: PERFORMANCE REPORTS
# ============================================
elif page == "Performance Reports":
    st.header("Detailed Performance Reports")
    
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Department Rankings", "Export Data"])
    
    with tab1:
        st.subheader("Model Performance Summary")
        
        # Summary table
        summary = results.groupby('Model').agg({
            'MAPE': ['mean', 'std', 'min', 'max']
        }).round(2)
        summary.columns = ['Mean MAPE', 'Std Dev', 'Best', 'Worst']
        st.dataframe(summary, use_container_width=True)
        
        # Box plot
        fig = px.box(results, x='Model', y='MAPE', color='Model',
                    title="Performance Distribution by Model")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Department Rankings")
        
        # Filter by model
        model_filter = st.selectbox("Select Model", ['All'] + list(results['Model'].unique()))
        
        if model_filter == 'All':
            display_data = results.copy()
        else:
            display_data = results[results['Model'] == model_filter].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Top 10 Best Performing")
            best = display_data.nsmallest(10, 'MAPE')[['Store', 'Dept', 'Model', 'MAPE']]
            st.dataframe(best, use_container_width=True)
        
        with col2:
            st.markdown("### Bottom 10 Worst Performing")
            worst = display_data.nlargest(10, 'MAPE')[['Store', 'Dept', 'Model', 'MAPE']]
            st.dataframe(worst, use_container_width=True)
    
    with tab3:
        st.subheader("Export Data")
        
        if st.button("Generate Full Report"):
            # Create summary report
            report = pd.DataFrame()
            report['Metric'] = ['Total Departments', 'Average MAPE', 'Best MAPE', 'Worst MAPE',
                               'Most Common Best Model', 'Total Annual Sales']
            report['Value'] = [
                len(results['Dept'].unique()),
                f"{results['MAPE'].mean():.1f}%",
                f"{results['MAPE'].min():.1f}%",
                f"{results['MAPE'].max():.1f}%",
                results.loc[results.groupby(['Store', 'Dept'])['MAPE'].idxmin()]['Model'].mode()[0],
                f"${df['Weekly_Sales'].sum()*52:,.0f}"
            ]
            
            st.dataframe(report, use_container_width=True)
            
            # Download buttons
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv,
                file_name="walmart_model_results.csv",
                mime="text/csv"
            )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Walmart Sales Forecasting System {datetime.now().strftime('%Y')}
    </div>
    """, 
    unsafe_allow_html=True

)

