import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import sin, cos, pi
import warnings
from tqdm import tqdm

def load_data(file_path="./../dataset/processed/selected_features.csv", site_id_path=None):
    """
    Load the dataset with selected features and handle missing Timestamp and SiteId columns.
    
    Args:
        file_path: Path to the selected features CSV
        site_id_path: Optional path to file containing Timestamp and SiteId
        
    Returns:
        DataFrame with all necessary columns
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        required_cols = ['Value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        has_timestamp = 'Timestamp' in df.columns
        has_site_id = 'SiteId' in df.columns
        
        if not has_timestamp or not has_site_id:
            if site_id_path:
                try:
                    site_df = pd.read_csv(site_id_path, parse_dates=["Timestamp"])
                    site_df = site_df.sort_values(["SiteId", "Timestamp"])
                    
                    for lag in [672]:
                        site_df[f'Value_lag_{lag}'] = site_df.groupby('SiteId')['Value'].shift(lag)
                    
                    site_df = site_df.dropna().reset_index(drop=True)
                    df = df.reset_index(drop=True)
                    
                    if len(site_df) != len(df):
                        print(f"Warning: Shape mismatch between selected features ({len(df)}) and site data ({len(site_df)})")
                        min_len = min(len(df), len(site_df))
                        df = df.iloc[:min_len]
                        site_df = site_df.iloc[:min_len]
                    
                    if not has_timestamp:
                        df['Timestamp'] = site_df['Timestamp']
                    if not has_site_id:
                        df['SiteId'] = site_df['SiteId']
                    
                except Exception as e:
                    print(f"Error loading site_id data: {str(e)}")
                    add_placeholder_columns(df, has_timestamp, has_site_id)
            else:
                add_placeholder_columns(df, has_timestamp, has_site_id)
                
        return df
    
    except Exception as e:
        print(f"Error during data loading: {str(e)}")
        print("Please check:")
        print("1. The file path to your CSV")
        print("2. Column names in your data")
        print("3. The datetime format if using timestamps")
        raise

def add_placeholder_columns(df, has_timestamp, has_site_id):
    """Add placeholder columns if real data is not available"""
    if not has_timestamp:
        print("Warning: No timestamp column found. Creating sequential timestamps.")
        df['Timestamp'] = pd.date_range(start='2014-01-01', periods=len(df), freq='H')
    
    if not has_site_id:
        print("Warning: No SiteId column found. Using default site ID.")
        if 'IsWeekend' in df.columns:
            df['SiteId'] = df.groupby('IsWeekend').ngroup() + 1
        else:
            df['SiteId'] = 1

def analyze_site_temporal_distribution(df, timestamp_col='Timestamp', site_col='SiteId'):
    """Analyze the temporal range of each site"""
    print("\nAnalyzing site temporal distribution...")
    site_ranges = df.groupby(site_col)[timestamp_col].agg(['min', 'max', 'count']).reset_index()
    site_ranges['duration'] = (site_ranges['max'] - site_ranges['min']).dt.total_seconds() / (3600 * 24)
    print(site_ranges.sort_values('count', ascending=False))
    
    short_duration = site_ranges[site_ranges['duration'] < 180]
    if not short_duration.empty:
        print(f"Warning: {len(short_duration)} sites have less than 6 months of data:")
        print(short_duration[[site_col, 'count', 'duration']])
    
    try:
        plt.figure(figsize=(12, 6))
        for site in site_ranges['SiteId']:
            site_data = site_ranges[site_ranges['SiteId'] == site]
            plt.hlines(y=site, xmin=site_data['min'], xmax=site_data['max'], linewidth=2)
        plt.title('Temporal Coverage by Site')
        plt.xlabel('Timestamp')
        plt.ylabel('SiteId')
        output_dir = './../reports'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/site_temporal_coverage.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create temporal coverage plot: {str(e)}")
    
    return site_ranges

def preprocess_data(df, normalize_by_site=True, min_temp_std=0.5, handle_outliers=True):
    """
    Preprocess the data with improved handling for features
    """
    print("\nPreprocessing data...")
    df = df.copy()
    
    if 'IsWeekend' in df.columns:
        df['IsWeekend'] = df['IsWeekend'].astype('category')
    
    # Add rolling mean features
    print("Adding rolling mean features...")
    for window in [24, 168]:  # 24 hours, 7 days
        df[f'Value_rolling_mean_{window}'] = df.groupby('SiteId')['Value'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Add day of week
    if 'Timestamp' in df.columns:
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df = df.drop('DayOfWeek', axis=1)
    
    # Drop unstable long lag features
    unstable_lags = ['Value_lag_672', 'Value_lag_384']
    df = df.drop([col for col in unstable_lags if col in df.columns], axis=1)
    
    # Handle outliers in value features
    value_features = [col for col in df.columns if 'Value' in col and col != 'Value']
    if handle_outliers:
        print("Handling outliers in value features...")
        for feature in value_features:
            if feature in df.columns:
                q1 = df[feature].quantile(0.01)
                q3 = df[feature].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers_count = ((df[feature] < lower_bound) | (df[feature] > upper_bound)).sum()
                if outliers_count > 0:
                    print(f"  Capping {outliers_count} outliers in {feature} ({outliers_count/len(df)*100:.2f}%)")
                df[feature] = df[feature].clip(lower_bound, upper_bound)
    
    # Global normalization
    print("Performing global normalization...")
    for feature in value_features:
        if feature in df.columns:
            mean = df[feature].mean()
            std = df[feature].std() or 1
            df[feature] = (df[feature] - mean) / std
    
    if 'Temperature' in df.columns:
        temp_std = df['Temperature'].std()
        print(f"Temperature standard deviation: {temp_std:.2f}")
        if temp_std < min_temp_std:
            print(f"Temperature has low variance (<{min_temp_std}°C), replacing with BaseTemperature.")
            if 'BaseTemperature' in df.columns:
                df['Temperature'] = df['BaseTemperature']
                print("Using BaseTemperature as Temperature.")
            else:
                print("Dropping Temperature due to low variance.")
                df = df.drop('Temperature', axis=1)
        else:
            print("Temperature has sufficient variance, keeping it.")
            
    if 'Hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df = df.drop('Hour', axis=1)
    
    if 'Month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df = df.drop('Month', axis=1)
    
    return df

def prepare_features_target(df, target_col='Value'):
    """Split features and target variable"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def identify_column_types(X):
    """Identify numerical and categorical columns"""
    categorical_cols = [col for col in X.columns if X[col].dtype.name == 'category']
    numerical_cols = [col for col in X.columns 
                     if col not in categorical_cols 
                     and col != 'SiteId' 
                     and col != 'Timestamp'
                     and X[col].dtype != 'object']
    return numerical_cols, categorical_cols

def create_preprocessing_pipeline(numerical_cols, categorical_cols, use_robust_scaler=True):
    """Create a preprocessing pipeline for scaling and encoding"""
    numerical_transformer = RobustScaler() if use_robust_scaler else StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    transformers = []
    if numerical_cols:
        transformers.append(('num', numerical_transformer, numerical_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    return ColumnTransformer(transformers=transformers)

def site_based_split(df, test_size=0.2, min_site_records=1000, min_site_overlap=0.5):
    """
    Split data based on sites to maximize overlap, filtering sparse sites
    """
    # Filter sites with sufficient data
    site_counts = df['SiteId'].value_counts()
    valid_sites = site_counts[site_counts >= min_site_records].index
    if len(valid_sites) < len(site_counts):
        print(f"Warning: Filtering {len(site_counts) - len(valid_sites)} sites with fewer than {min_site_records} records.")
        df = df[df['SiteId'].isin(valid_sites)].copy()
    
    if len(valid_sites) < 2:
        raise ValueError("Fewer than 2 sites with sufficient records. Adjust min_site_records or check data.")
    
    # Sort by Timestamp within each site
    df = df.sort_values(['SiteId', 'Timestamp'])
    
    # Perform stratified split by SiteId
    train_dfs = []
    test_dfs = []
    for site in valid_sites:
        site_data = df[df['SiteId'] == site]
        if len(site_data) < 2:
            continue
        train_site, test_site = train_test_split(site_data, test_size=test_size, shuffle=False)
        train_dfs.append(train_site)
        test_dfs.append(test_site)
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    
    # Check site overlap
    train_sites = set(train_df['SiteId'].unique())
    test_sites = set(test_df['SiteId'].unique())
    overlap = len(train_sites.intersection(test_sites)) / max(1, len(train_sites))
    
    if overlap < min_site_overlap:
        print(f"Warning: Site overlap ({overlap:.1%}) below threshold ({min_site_overlap:.1%}).")
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Split resulted in empty train or test set. Check data distribution.")
    
    return train_df, test_df

def split_and_scale_data(df, test_size=0.2, timestamp_col='Timestamp', 
                        target_col='Value', use_robust_scaler=True):
    """
    Split data using site-based approach and preprocess features
    """
    print(f"\nPerforming site-based split with test size {test_size}")
    
    train_df, test_df = site_based_split(
        df, 
        test_size=test_size, 
        min_site_records=1000,
        min_site_overlap=0.5
    )
    
    print("\nSite distribution analysis:")
    train_site_counts = train_df['SiteId'].value_counts()
    test_site_counts = test_df['SiteId'].value_counts()
    train_sites = set(train_df['SiteId'].unique())
    test_sites = set(test_df['SiteId'].unique())
    common_sites = train_sites.intersection(test_sites)
    
    print(f"Sites in train: {len(train_sites)}")
    print(f"Sites in test: {len(test_sites)}")
    print(f"Common sites: {len(common_sites)} ({len(common_sites)/max(1, len(train_sites)):.1%} of train sites)")
    
    if len(common_sites) < 0.5 * len(train_sites):
        print("⚠️ WARNING: Less than 50% site overlap. Consider reducing min_site_records.")
    
    try:
        plt.figure(figsize=(10, 6))
        train_sample = train_df.sample(min(1000, len(train_df)))
        test_sample = test_df.sample(min(1000, len(test_df)))
        plt.scatter(train_sample[timestamp_col], train_sample[target_col], 
                   alpha=0.5, label='Training data', color='blue')
        plt.scatter(test_sample[timestamp_col], test_sample[target_col], 
                   alpha=0.5, label='Test data', color='red')
        plt.legend()
        plt.title('Site-based Train-Test Split')
        plt.xlabel('Timestamp')
        plt.ylabel(target_col)
        output_dir = './../reports'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/site_split_visualization.png')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create split visualization: {str(e)}")
    
    print(f"\nTrain set size: {len(train_df)}, Test set size: {len(test_df)}")
    
    train_timestamps = train_df[timestamp_col].copy()
    test_timestamps = test_df[timestamp_col].copy()
    train_site_ids = train_df['SiteId'].copy()
    test_site_ids = test_df['SiteId'].copy()
    
    train_df = train_df.drop([timestamp_col, 'SiteId'], axis=1)
    test_df = test_df.drop([timestamp_col, 'SiteId'], axis=1)
    
    X_train, y_train = prepare_features_target(train_df, target_col)
    X_test, y_test = prepare_features_target(test_df, target_col)
    
    numerical_cols, categorical_cols = identify_column_types(X_train)
    
    print(f"\nNumerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols, 
                                               use_robust_scaler=use_robust_scaler)
    print("\nFitting preprocessing pipeline on training data...")
    preprocessor.fit(X_train)
    
    print("Transforming training and test data...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols)
    
    check_distribution_shift(X_train, X_test, numerical_cols)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_transformed': X_train_transformed,
        'X_test_transformed': X_test_transformed,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'train_timestamps': train_timestamps,
        'test_timestamps': test_timestamps,
        'train_site_ids': train_site_ids,
        'test_site_ids': test_site_ids
    }

def check_distribution_shift(X_train, X_test, numerical_cols, max_features=5, significant_shift_pct=10):
    """
    Check for distribution shift between train and test sets
    """
    print("\nChecking for distribution shift in key features...")
    
    features_to_check = [col for col in numerical_cols if 'Value' in col][:max_features]
    shift_detected = False
    output_dir = './../reports'
    os.makedirs(output_dir, exist_ok=True)
    shift_metrics = []
    
    for feature in features_to_check:
        if feature in X_train.columns and feature in X_test.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.kdeplot(X_train[feature], label='Train', fill=True, alpha=0.3)
                sns.kdeplot(X_test[feature], label='Test', fill=True, alpha=0.3)
                plt.title(f'Distribution Shift: {feature}')
                plt.legend()
                plt.savefig(f'{output_dir}/distribution_{feature}.png')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create distribution plot for {feature}: {str(e)}")
            
            train_mean = X_train[feature].mean()
            test_mean = X_test[feature].mean()
            train_median = X_train[feature].median()
            test_median = X_test[feature].median()
            
            mean_diff_pct = abs(train_mean - test_mean) / max(abs(train_mean), 0.001) * 100
            median_diff_pct = abs(train_median - test_median) / max(abs(train_median), 0.001) * 100
            
            print(f"  {feature}:")
            print(f"    Mean shift: {mean_diff_pct:.1f}% (Train={train_mean:.3f}, Test={test_mean:.3f})")
            print(f"    Median shift: {median_diff_pct:.1f}% (Train={train_median:.3f}, Test={test_median:.3f})")
            
            shift_metrics.append({
                'Feature': feature,
                'Mean_Shift_Pct': mean_diff_pct,
                'Median_Shift_Pct': median_diff_pct
            })
            
            if mean_diff_pct > significant_shift_pct or median_diff_pct > significant_shift_pct:
                print(f"    ⚠️ WARNING: Distribution shift detected in {feature}")
                shift_detected = True
    
    pd.DataFrame(shift_metrics).to_csv(f'{output_dir}/distribution_shift_metrics.csv', index=False)
    
    if shift_detected:
        print("\n⚠️ DISTRIBUTION SHIFT DETECTED: Consider these mitigation strategies:")
        print("1. Use more robust features (e.g., shorter rolling windows)")
        print("2. Validate with cross-validation across sites")
        print("3. Add more site-invariant features (e.g., holidays)")
        print("4. Check data quality for site-specific anomalies")

def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    """Get feature names after transformation"""
    feature_names = []
    try:
        if numerical_cols:
            feature_names.extend(numerical_cols)
        if categorical_cols:
            ohe_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols)
            feature_names.extend(list(ohe_feature_names))
    except Exception as e:
        print(f"Warning: Error extracting feature names: {str(e)}")
        feature_names = [f"feature_{i}" for i in range(preprocessor.transform(pd.DataFrame({col: [0] for col in numerical_cols + categorical_cols})).shape[1])]
    return feature_names

def save_processed_data(result_dict, output_dir='./../models'):
    """Save processed data and preprocessing objects"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/preprocessor.pkl", "wb") as f:
            pickle.dump(result_dict['preprocessor'], f)
        
        np.save(f"{output_dir}/X_train_transformed.npy", result_dict['X_train_transformed'])
        np.save(f"{output_dir}/X_test_transformed.npy", result_dict['X_test_transformed'])
        np.save(f"{output_dir}/y_train.npy", result_dict['y_train'].values)
        np.save(f"{output_dir}/y_test.npy", result_dict['y_test'].values)
        
        with open(f"{output_dir}/feature_names.pkl", "wb") as f:
            pickle.dump(result_dict['feature_names'], f)
        
        result_dict['X_train'].to_csv(f"{output_dir}/X_train_original.csv", index=False)
        result_dict['X_test'].to_csv(f"{output_dir}/X_test_original.csv", index=False)
        
        pd.DataFrame({'y_train': result_dict['y_train']}).to_csv(f"{output_dir}/y_train_values.csv", index=False)
        pd.DataFrame({'y_test': result_dict['y_test']}).to_csv(f"{output_dir}/y_test_values.csv", index=False)
        
        if 'train_timestamps' in result_dict:
            pd.DataFrame({
                'train_timestamps': result_dict['train_timestamps'],
                'train_site_ids': result_dict['train_site_ids']
            }).to_csv(f"{output_dir}/train_metadata.csv", index=False)
            
            pd.DataFrame({
                'test_timestamps': result_dict['test_timestamps'],
                'test_site_ids': result_dict['test_site_ids']
            }).to_csv(f"{output_dir}/test_metadata.csv", index=False)
        
        print(f"\n✅ Processed data saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error saving processed data: {str(e)}")
        raise

def analyze_feature_importance(result_dict):
    """Analyze feature importance using correlation with target"""
    print("\nAnalyzing feature importance...")
    
    X_train = result_dict['X_train'].copy()
    X_train['target'] = result_dict['y_train'].values
    
    correlations = X_train.corr()['target'].drop('target').abs().sort_values(ascending=False)
    
    print("\nTop features by correlation with target:")
    print(correlations.head(10))
    
    plt.figure(figsize=(10, 8))
    top_n = 15
    top_corr = correlations.head(min(top_n, len(correlations)))
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title(f'Top {len(top_corr)} Features by Correlation with Target')
    plt.tight_layout()
    output_dir = './../reports'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    return correlations

def create_feature_summary(correlations, result_dict):
    """Create a feature summary file with separate handling for numerical and categorical columns"""
    print("\nCreating feature summary...")
    
    numerical_cols, categorical_cols = identify_column_types(result_dict['X_train'])
    features = result_dict['X_train'].columns
    corr_values = correlations.reindex(features).values
    train_means = []
    train_stds = []
    test_means = []
    test_stds = []
    
    for col in features:
        if col in numerical_cols:
            train_means.append(result_dict['X_train'][col].mean())
            train_stds.append(result_dict['X_train'][col].std())
            test_means.append(result_dict['X_test'][col].mean())
            test_stds.append(result_dict['X_test'][col].std())
        else:
            train_means.append(np.nan)
            train_stds.append(np.nan)
            test_means.append(np.nan)
            test_stds.append(np.nan)
    
    summary = pd.DataFrame({
        'Feature': features,
        'Correlation_with_Target': corr_values,
        'Train_Mean': train_means,
        'Train_Std': train_stds,
        'Test_Mean': test_means,
        'Test_Std': test_stds,
    })
    
    summary['Mean_Shift_Pct'] = (
        abs(summary['Train_Mean'] - summary['Test_Mean']) / 
        summary['Train_Mean'].abs().replace(0, 0.001) * 100
    )
    
    summary = summary.sort_values('Correlation_with_Target', ascending=False)
    
    output_dir = './../reports'
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(f'{output_dir}/feature_summary.csv', index=False)
    print(f"\n✅ Feature summary saved to {output_dir}/feature_summary.csv")
    
    return summary

def validate_pipeline(result_dict):
    """Validate the preprocessing pipeline output"""
    try:
        assert result_dict['X_train_transformed'].shape[0] == len(result_dict['y_train']), \
            "Mismatch in training data size"
        assert result_dict['X_test_transformed'].shape[0] == len(result_dict['y_test']), \
            "Mismatch in test data size"
        assert not np.any(np.isnan(result_dict['X_train_transformed'])), \
            "NaN values in transformed training data"
        assert not np.any(np.isnan(result_dict['X_test_transformed'])), \
            "NaN values in transformed test data"
        print("✅ Pipeline validation passed!")
    except AssertionError as e:
        print(f"❌ Pipeline validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    sample_data_path = "./../dataset/processed/sample_data.csv"
    selected_features_path = "./../dataset/processed/selected_features.csv"
    
    try:
        df = load_data(selected_features_path, site_id_path=sample_data_path)
        print("\nDataset shape:", df.shape)
        print("Feature data types:")
        print(df.dtypes)
        
        site_counts = df['SiteId'].value_counts()
        print("\nSite distribution before split:")
        print(site_counts.head())
        if site_counts.min() < 1000:
            print("⚠️ Warning: Some sites have very few records. Consider filtering.")
        
        site_ranges = analyze_site_temporal_distribution(df)
        
        df = preprocess_data(df, normalize_by_site=False, min_temp_std=0.5, handle_outliers=True)
        print("\nAfter preprocessing - Dataset shape:", df.shape)
        print("After preprocessing - Feature data types:")
        print(df.dtypes)
        
        result_dict = split_and_scale_data(
            df, 
            test_size=0.2,
            use_robust_scaler=True
        )
        
        validate_pipeline(result_dict)
        
        correlations = analyze_feature_importance(result_dict)
        summary = create_feature_summary(correlations, result_dict)
        
        save_processed_data(result_dict)
        
        print("\n🚀 Ready for model training!")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")