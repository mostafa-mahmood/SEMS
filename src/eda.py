# EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import calendar
from scipy import stats
import sys

# Set global styles
sns.set(style="whitegrid", palette="pastel")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 300

def format_kwh(x, pos):
    """Formatter function for kWh values"""
    return f"{x/1000:,.0f}k" if x >= 10000 else f"{x:,.0f}"

kwh_formatter = FuncFormatter(format_kwh)

def clean_time_series(series):
    """Handle missing values in time series data"""
    # Forward fill then backward fill small gaps
    series = series.ffill().bfill()
    
    # For larger gaps, consider linear interpolation
    if series.isnull().sum() > 0:
        series = series.interpolate(method='time')
    
    return series

def perform_enhanced_eda(df, output_dir="./../reports"):
    """
    Comprehensive EDA for Smart Energy Management System
    
    Parameters:
        df (pd.DataFrame): Input dataframe with energy data
        output_dir (str): Directory to save visualizations
    """
    # Convert data types
    df['SiteId'] = df['SiteId'].astype('category')
    df['IsWeekend'] = df['IsWeekend'].astype(bool)
    df['IsHoliday'] = df['IsHoliday'].astype(bool)
    
    # Create time-based features
    df['Year'] = df['Timestamp'].dt.year
    df['DayOfYear'] = df['Timestamp'].dt.dayofyear
    df['WeekdayName'] = df['Timestamp'].dt.day_name()
    df['Season'] = df['Timestamp'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Print basic information
    print("🔍 Dataset Overview:")
    print(f"Time Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Total Observations: {len(df):,}")
    print(f"Unique Buildings: {df['SiteId'].nunique()}")
    print(f"Recording Frequency: {(df['Timestamp'].diff().mode()[0]).total_seconds()/60} minutes")
    
    # 1. Energy Consumption Distribution Analysis
    plt.figure(figsize=(18, 6))
    
    # Raw values
    plt.subplot(1, 3, 1)
    sns.histplot(df['Value'], bins=50, kde=True)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.title("Raw Energy Consumption Distribution")
    plt.xlabel("Energy Consumption (units)")
    
    # Log-transformed
    plt.subplot(1, 3, 2)
    log_values = np.log1p(df['Value'])
    sns.histplot(log_values, bins=50, kde=True)
    plt.title("Log-Transformed Distribution")
    plt.xlabel("log(Energy Consumption + 1)")
    
    # QQ plot
    plt.subplot(1, 3, 3)
    stats.probplot(log_values, plot=plt)
    plt.title("Q-Q Plot of Log-Transformed Values")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/consumption_distributions.png", bbox_inches='tight')
    plt.show()
    
    # 2. Temporal Patterns Analysis
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Hourly pattern (with observed=True to fix warning)
    hourly = df.groupby('Hour', observed=True)['Value'].mean()
    sns.lineplot(x=hourly.index, y=hourly.values, ax=axes[0])
    axes[0].set_title("Average Hourly Energy Consumption Pattern")
    axes[0].set_ylabel("Average Consumption")
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(24))
    
    # Weekly pattern (with observed=True)
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = df.groupby('WeekdayName', observed=True)['Value'].mean().reindex(weekday_order)
    sns.barplot(x=weekly.index, y=weekly.values, ax=axes[1])
    axes[1].set_title("Average Weekly Energy Consumption Pattern")
    axes[1].set_ylabel("Average Consumption")
    
    # Monthly pattern (with observed=True)
    monthly = df.groupby('Month', observed=True)['Value'].mean()
    sns.lineplot(x=monthly.index, y=monthly.values, ax=axes[2], marker='o')
    axes[2].set_title("Average Monthly Energy Consumption Pattern")
    axes[2].set_ylabel("Average Consumption")
    axes[2].set_xticks(range(1,13))
    axes[2].set_xticklabels([calendar.month_abbr[i] for i in range(1,13)])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_patterns.png", bbox_inches='tight')
    plt.show()
    
    # 3. Building Comparison Analysis
    plt.figure(figsize=(14, 8))
    
    # Order buildings by median consumption (with observed=True)
    building_order = df.groupby('SiteId', observed=True)['Value'].median().sort_values().index
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, y='SiteId', x='Value', order=building_order, showfliers=False)
    plt.title("Energy Consumption Distribution by Building\n(Excluding Outliers)")
    plt.xlabel("Energy Consumption")
    plt.gca().xaxis.set_major_formatter(kwh_formatter)
    
    plt.subplot(1, 2, 2)
    building_hourly = df.groupby(['SiteId', 'Hour'], observed=True)['Value'].mean().reset_index()
    sns.lineplot(data=building_hourly, x='Hour', y='Value', hue='SiteId', 
                 palette='tab20', legend='full')
    plt.title("Hourly Consumption Patterns by Building")
    plt.ylabel("Average Consumption")
    plt.gca().yaxis.set_major_formatter(kwh_formatter)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/building_comparison.png", bbox_inches='tight')
    plt.show()
    
    # 4. Temperature Impact Analysis
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df.sample(1000), x='Temperature', y='Value', alpha=0.6)
    plt.title("Energy Consumption vs Temperature")
    plt.ylabel("Consumption")
    plt.gca().yaxis.set_major_formatter(kwh_formatter)
    
    plt.subplot(1, 2, 2)
    temp_bins = pd.cut(df['Temperature'], bins=20)
    temp_impact = df.groupby(temp_bins, observed=True)['Value'].mean().reset_index()
    temp_impact['Temperature'] = temp_impact['Temperature'].apply(lambda x: x.mid)
    sns.regplot(data=temp_impact, x='Temperature', y='Value', 
                order=2, ci=None, scatter_kws={'s': 100})
    plt.title("Binned Average Consumption by Temperature")
    plt.ylabel("Average Consumption")
    plt.gca().yaxis.set_major_formatter(kwh_formatter)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temperature_impact.png", bbox_inches='tight')
    plt.show()
    
    # 5. Special Day Analysis
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='IsWeekend', y='Value')
    plt.title("Weekend vs Weekday Consumption")
    plt.xlabel("Is Weekend?")
    plt.ylabel("Energy Consumption")
    plt.gca().yaxis.set_major_formatter(kwh_formatter)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='IsHoliday', y='Value')
    plt.title("Holiday vs Non-Holiday Consumption")
    plt.xlabel("Is Holiday?")
    plt.ylabel("Energy Consumption")
    plt.gca().yaxis.set_major_formatter(kwh_formatter)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/special_days.png", bbox_inches='tight')
    plt.show()
    
    # 6. Correlation Analysis
    numeric_cols = ['Value', 'Hour', 'DayOfWeek', 'Month', 'Surface', 
                   'BaseTemperature', 'Temperature', 'Distance']
    
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", bbox_inches='tight')
    plt.show()
    
    # 7. Time Series Decomposition (with missing value handling)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        example_site = df['SiteId'].value_counts().index[0]
        site_df = df[df['SiteId'] == example_site].set_index('Timestamp').resample('D')['Value'].mean()
        
        # Clean the time series before decomposition
        site_df_clean = clean_time_series(site_df)
        
        if site_df_clean.isnull().sum() > 0:
            print(f"Warning: {site_df_clean.isnull().sum()} missing values remain after cleaning - decomposition may fail", 
                  file=sys.stderr)
        
        result = seasonal_decompose(site_df_clean, model='additive', period=365)
        
        plt.figure(figsize=(14, 10))
        result.plot()
        plt.suptitle(f"Time Series Decomposition for Site {example_site}", y=1.02)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/time_series_decomposition.png", bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("statsmodels not installed - skipping time series decomposition", file=sys.stderr)
    except ValueError as e:
        print(f"Could not perform decomposition: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error during decomposition: {e}", file=sys.stderr)

# Load and process data
df = pd.read_csv("./../dataset/processed/sample_data.csv", parse_dates=["Timestamp"])
perform_enhanced_eda(df)