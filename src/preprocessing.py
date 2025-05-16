import pandas as pd
import os
import numpy as np
from datetime import datetime
import json

# To merge all CSV's into one file
def merge_data():
    print("Starting data merging...")
    
    # Define file paths
    base_path = "./../dataset/raw/"
    metadata_path = os.path.join(base_path, "metadata.csv")
    weather_path = os.path.join(base_path, "weather.csv")
    holidays_path = os.path.join(base_path, "holidays.csv")
    train_data_path = os.path.join(base_path, "train-data.csv")
    
    # Load all datasets with proper separator
    print("Loading datasets...")
    metadata_df = pd.read_csv(metadata_path, sep=';')
    weather_df = pd.read_csv(weather_path, sep=';')
    holidays_df = pd.read_csv(holidays_path, sep=';')
    train_df = pd.read_csv(train_data_path, sep=';')
    
    print(f"Loaded train data with {len(train_df)} records")
    
    # Convert timestamps to datetime objects
    print("Converting timestamps...")
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    weather_df['Timestamp'] = pd.to_datetime(weather_df['Timestamp'])
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])
    
    # Extract time features from timestamp
    print("Extracting time features...")
    train_df['Hour'] = train_df['Timestamp'].dt.hour
    train_df['DayOfWeek'] = train_df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    train_df['IsWeekend'] = train_df['DayOfWeek'].apply(lambda x: x >= 5)
    train_df['Month'] = train_df['Timestamp'].dt.month
    train_df['Day'] = train_df['Timestamp'].dt.day
    train_df['WeekOfYear'] = train_df['Timestamp'].dt.isocalendar().week
    
    # Merge train data with metadata
    print("Merging with metadata...")
    merged_df = pd.merge(train_df, metadata_df, on='SiteId', how='left')
    
    # Process weather data
    print("Merging with weather data...")
    # Find the closest weather station reading for each site and timestamp
    weather_closest = weather_df.sort_values('Distance').groupby(['SiteId', 'Timestamp']).first().reset_index()
    
    # Merge with weather data on both SiteId and Timestamp
    merged_df = pd.merge(
        merged_df, 
        weather_closest[['SiteId', 'Timestamp', 'Temperature', 'Distance']], 
        on=['SiteId', 'Timestamp'], 
        how='left'
    )
    
    # Fill missing weather data with the most recent available data for each site
    merged_df = merged_df.sort_values(['SiteId', 'Timestamp'])
    merged_df[['Temperature', 'Distance']] = merged_df.groupby('SiteId')[['Temperature', 'Distance']].fillna(method='ffill')
    
    # Process holidays data
    print("Processing holidays data...")
    # Extract date from timestamp for joining with holidays
    merged_df['Date'] = merged_df['Timestamp'].dt.date
    holidays_df['Date'] = holidays_df['Date'].dt.date
    
    # Create IsHoliday column
    # Create a set of (SiteId, Date) tuples for faster lookup
    holiday_set = set(zip(holidays_df['SiteId'], holidays_df['Date']))
    
    # Create IsHoliday column
    merged_df['IsHoliday'] = merged_df.apply(
        lambda row: (row['SiteId'], row['Date']) in holiday_set,
        axis=1
    )
    
    # Drop the temporary Date column used for holiday merging
    merged_df.drop('Date', axis=1, inplace=True)
    
    # Fill any remaining NaN values
    print("Handling missing values...")
    # For weather data, use site averages where available
    site_avg_temp = merged_df.groupby('SiteId')['Temperature'].transform('mean')
    merged_df['Temperature'].fillna(site_avg_temp, inplace=True)
    
    # For any remaining NaN values, use global averages or reasonable defaults
    merged_df['Temperature'].fillna(merged_df['Temperature'].mean(), inplace=True)
    merged_df['Distance'].fillna(merged_df['Distance'].mean(), inplace=True)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    output_path = "./../dataset/processed/merged_data.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    merged_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Data saved to {output_path}")
    print(f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    
    # Print sample of preprocessed data
    print("\nSample of merged data:")
    print(merged_df.head())
    
    return merged_df


import pandas as pd
import os
import json

def sample_data(input_path="./../dataset/processed/merged_data.csv", output_path="./../dataset/processed/sample_data.csv"):
    """
    Creates a representative sample of the energy consumption data that's manageable
    for ML model training, focusing on complete data for a subset of buildings.
    """
    print("Loading merged data...")
    df = pd.read_csv(input_path)
    print(f"Original dataset: {len(df)} records across {df['SiteId'].nunique()} buildings")
    
    # Convert timestamp back to datetime for time-based sampling
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Strategy: Select buildings with the most complete data
    # First, check data completeness by building
    completeness = df.groupby('SiteId').apply(
        lambda x: (x['Value'].notna().sum() / len(x)) * 100
    ).sort_values(ascending=False)
    
    print(f"Top 5 buildings by data completeness (%): \n{completeness.head()}")
    
    # Select the top 50 buildings with most complete data (increased from 15)
    top_buildings = completeness.head(50).index.tolist()
    
    # Filter for these buildings
    building_sample = df[df['SiteId'].isin(top_buildings)]
    print(f"Selected {len(building_sample)} records from {len(top_buildings)} buildings")
    
    # For each building, take at most 6 months of data to further reduce size if needed
    if len(building_sample) > 150000:  # Lowered threshold from 500,000
        print("Further reducing sample size by taking 6 months of data per building...")
        
        final_sample = pd.DataFrame()
        for site_id in top_buildings:
            site_data = building_sample[building_sample['SiteId'] == site_id]
            
            # Sort by time
            site_data = site_data.sort_values('Timestamp')
            
            # Get the date range
            start_date = site_data['Timestamp'].min()
            
            # Take 6 months (180 days) of data (increased from 90 days)
            end_date = start_date + pd.Timedelta(days=180)
            time_window = site_data[
                (site_data['Timestamp'] >= start_date) & 
                (site_data['Timestamp'] <= end_date)
            ]
            
            # Add to our final sample
            final_sample = pd.concat([final_sample, time_window])
            
        building_sample = final_sample
        print(f"Reduced to {len(building_sample)} records covering 6 months per building")
    
    # Check data quality of the sample
    missing_values = building_sample.isna().sum()
    print("\nMissing values in the sample:")
    print(missing_values[missing_values > 0])
    
    # Handle any remaining missing values in the Value column
    if building_sample['Value'].isna().sum() > 0:
        print("Filling missing Values with median for respective buildings and hours...")
        # Group by SiteId and Hour, then fill missing values with the median for that group
        building_sample['Value'] = building_sample.groupby(['SiteId', 'Hour'])['Value'].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else x.mean())
        )
    
    # Final verification - ensure no missing values in key columns
    for col in ['Value', 'Temperature', 'Distance']:
        if building_sample[col].isna().sum() > 0:
            building_sample[col] = building_sample[col].fillna(building_sample[col].median())
    
    print(f"\nFinal sample: {len(building_sample)} records from {building_sample['SiteId'].nunique()} buildings")
    print(f"Date range: {building_sample['Timestamp'].min()} to {building_sample['Timestamp'].max()}")
    
    # Save the sample
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    building_sample.to_csv(output_path, index=False)
    print(f"Sample saved to {output_path}")
    
    site_counts = building_sample['SiteId'].value_counts()

    sample_meta = {
        "num_records": len(building_sample),
        "num_buildings": building_sample['SiteId'].nunique(),
        "start_date": str(building_sample['Timestamp'].min()),
        "end_date": str(building_sample['Timestamp'].max()),
        "site_counts": site_counts.to_dict()
    }
    with open("./../dataset/processed/sample_metadata.json", "w") as f:
        json.dump(sample_meta, f, indent=4)
    print("Sample metadata saved to dataset/processed/sample_metadata.json")
    # Provide distribution info for reference
    print("\nSample data distribution by building:")
    
    for site_id, count in site_counts.items():
        print(f"Building {site_id}: {count} records")
    
    return building_sample

# if __name__ == "__main__":
#     sample_data()
