import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv("./../dataset/processed/sample_data.csv", parse_dates=["Timestamp"])
    
    # Sort by SiteId and Timestamp to ensure chronological order
    df = df.sort_values(["SiteId", "Timestamp"])
    
    # Ensure categorical types
    categorical_cols = ['SiteId', 'IsWeekend', 'IsHoliday']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Drop any duplicate rows that could cause leakage
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    return df

def create_lag_features(df, lags=[20, 40, 48, 192, 384, 672]):
    """
    Create lag features for the target variable within each site
    Lags represent:
    - 20: 5 hours ago (20 * 15 minutes)
    - 40: 10 hours ago
    - 48: 12 hours ago
    - 192: 2 days ago (48 hours)
    - 384: 4 days ago
    - 672: 1 week ago (7 days)
    """
    df = df.copy()
    
    # Ensure chronological order within each site
    df = df.sort_values(['SiteId', 'Timestamp'])
    
    # Create lag features for each site
    for lag in lags:
        # Group by SiteId and shift the Value column, suppress FutureWarning
        df[f'Value_lag_{lag}'] = df.groupby('SiteId', observed=False)['Value'].shift(lag)
    
    # Debug: Print sample rows for SiteId=5
    print("Sample rows for SiteId=5 after creating lag features:")
    site_5 = df[df['SiteId'] == 5][['Timestamp', 'Value', 'Value_lag_20', 'Value_lag_192']].head(5)
    print(site_5.to_string(index=False))
    
    # Debug: Check sites with constant values
    variance_per_site = df.groupby('SiteId', observed=True)['Value'].var()
    constant_sites = variance_per_site[variance_per_site == 0].index.tolist()
    if constant_sites:
        print(f"Sites with constant Value: {constant_sites}")
        for site in constant_sites[:2]:  # Limit to first 2 for brevity
            print(f"Sample rows for SiteId={site}:")
            site_data = df[df['SiteId'] == site][['Timestamp', 'Value', 'Value_lag_20', 'Value_lag_192']].head(5)
            print(site_data.to_string(index=False))
    
    # Drop rows with NaN lag values
    rows_before = len(df)
    df = df.dropna()
    rows_removed = rows_before - len(df)
    print(f"Removed {rows_removed} rows with missing lag values")
    
    return df

def analyze_correlations(df, target='Value', correlation_threshold=0.1):
    """Generate and analyze correlation matrix"""
    # Calculate correlations for numeric columns only
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Display correlations with the target variable
    print("Correlations with target 'Value':")
    print(corr_matrix[target].sort_values(ascending=False))
    
    # Check for features with perfect correlation (excluding the target itself)
    features_with_perfect_corr = corr_matrix[target][(abs(corr_matrix[target]) == 1.0) & (corr_matrix.index != target)].index.tolist()
    if features_with_perfect_corr:
        print(f"WARNING: These features have perfect correlation with target: {features_with_perfect_corr}")
        print("This might indicate data leakage or an error in feature creation.")
    else:
        print("No features have perfect correlation with target besides itself.")
    
    # Plot a focused correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix[[target]], annot=True, cmap='coolwarm', center=0,
                cbar_kws={'label': 'Correlation Strength'}, fmt='.3f')
    plt.title(f"Feature Correlations with {target}", pad=20)
    plt.tight_layout()
    plt.savefig("./../reports/feature_correlations.png")
    plt.close()
    
    # Select features with correlation above the threshold
    corr_features = corr_matrix[target][abs(corr_matrix[target]) > correlation_threshold].index.tolist()
    print(f"Features with correlation > {correlation_threshold}: {corr_features}")
    
    return corr_features

def select_features(df, target='Value', correlation_threshold=0.1):
    """Automated feature selection with domain knowledge integration"""
    # Step 1: Correlation-based feature selection
    corr_features = analyze_correlations(df, target, correlation_threshold)
    
    # Step 2: Include features based on domain knowledge
    domain_features = [
        'IsWeekend',      # Captures operational patterns (weekend vs weekday)
        'Hour',           # Hourly temporal patterns
        'Month',          # Seasonal patterns
        'Temperature',    # External environmental factor
        'BaseTemperature' # Building-specific setting
    ]
    
    # Include all lag features as they’re critical for time series
    lag_features = [col for col in df.columns if col.startswith('Value_lag_')]
    domain_features.extend(lag_features)
    
    # Step 3: Exclude irrelevant or redundant columns
    blacklist = [
        'obs_id',       # Unique identifier, not a predictor
        'ForecastId',   # Metadata, not a feature
        'Timestamp',    # Already extracted into Hour, Month, etc.
        'DayOfYear',    # Redundant with Month
        'WeekOfYear'    # Redundant with Month
    ]
    
    # Combine correlation-based and domain features, remove blacklisted ones
    selected = list(set(corr_features + domain_features))
    selected = [f for f in selected if f not in blacklist and f != target]
    
    # Create the final dataframe with target and selected features
    result_df = df[[target] + selected].copy()
    
    print(f"Selected {len(selected)} features: {sorted(selected)}")
    
    return result_df

def save_selected_features(df, path):
    """Save the processed dataset with selected features"""
    df.to_csv(path, index=False)
    print(f"✅ Selected {len(df.columns)} features saved to {path}")
    print("Selected features:", sorted(df.columns.tolist()))

if __name__ == "__main__":
    # Load the dataset
    df = load_data()
    print(f"Original dataset shape: {df.shape}")
    
    # Add lag features for time series prediction
    df = create_lag_features(df)
    print(f"Dataset shape after adding lag features: {df.shape}")
    
    # Perform feature selection
    selected_df = select_features(df, correlation_threshold=0.1)
    
    # Save the results
    save_selected_features(selected_df, "./../dataset/processed/selected_features.csv")
    
    # Generate a final correlation report for the selected features
    print("\nFinal Feature Correlations:")
    final_corr = analyze_correlations(selected_df, correlation_threshold=0.1)