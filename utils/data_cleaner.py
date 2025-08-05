from typing import List, Optional, Union, Dict, Tuple
import pandas as pd
import numpy as np
import json

def handle_datetime_missing(df: pd.DataFrame, 
                          exclude_columns: Optional[List[str]] = None,
                          inplace: bool = False) -> pd.DataFrame:
    
    if exclude_columns is None:
        exclude_columns = []
    
    # Work on copy unless inplace=True
    if not inplace:
        df = df.copy()
    
    # Store original row count
    original_rows = len(df)
    
    # Detect datetime columns
    datetime_columns = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
    
    # Filter out excluded columns
    columns_to_check = [col for col in datetime_columns if col not in exclude_columns]
    
    print(f"Detected datetime columns: {datetime_columns}")
    if exclude_columns:
        print(f"Excluded columns: {exclude_columns}")
    print(f"Checking for missing values in: {columns_to_check}")
    
    if not columns_to_check:
        print("No datetime columns to check for missing values.")
        return df
    
    # Check for missing values in each datetime column
    missing_info = {}
    for col in columns_to_check:
        missing_count = df[col].isna().sum()
        missing_info[col] = missing_count
        if missing_count > 0:
            print(f"Column '{col}': {missing_count} missing values found")
    
    # Remove rows with missing values in any of the datetime columns to check
    if columns_to_check:
        df_cleaned = df.dropna(subset=columns_to_check)
    else:
        df_cleaned = df
    
    # Calculate and report rows removed
    rows_removed = original_rows - len(df_cleaned)
    print(f"\nRows removed: {rows_removed}")
    print(f"Original rows: {original_rows}")
    print(f"Remaining rows: {len(df_cleaned)}")
    
    if inplace:
        df.drop(df.index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        for idx, row in df_cleaned.iterrows():
            df.loc[idx] = row
        return df
    else:
        return df_cleaned.reset_index(drop=True)
    

############################################################################################
# Outlier handler
############################################################################################
def read_outlier_file(table_name: str, file_path: str = '../config/outliers.json') -> Tuple[Dict, Union[str, list]]:
    """
    Supports both dict format {"min": x, "max": y} and list format [min, max].
    Default outlier file path is '../config/outliers.json'.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Outlier file not found: {file_path}")
        return {}, None
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON in outlier file: {file_path}")
        return {}, None
    
    table_mapping = {
        'labs': ('lab_category', 'lab_category'),
        'respiratory_support': ('respiratory_support', None),  # columns are keys
        'vitals': ('vital_category', 'vital_category')
    }
    
    if table_name not in table_mapping:
        print(f"⚠️ Table '{table_name}' not found in outlier configuration")
        return {}, None
    
    config_key, category_column = table_mapping[table_name]
    outlier_dict = data.get(config_key, {})
    
    # Convert dict format to list format for consistency
    for key, value in outlier_dict.items():
        if isinstance(value, dict) and 'min' in value and 'max' in value:
            outlier_dict[key] = [value['min'], value['max']]
        elif isinstance(value, list) and len(value) == 2:
            # Already in list format
            pass
        else:
            print(f"⚠️ Invalid format for '{key}'. Expected dict with min/max or list [min, max]")
            outlier_dict[key] = [0, float('inf')]  # Default safe values
    
    if table_name == 'respiratory_support':
        category_column = list(outlier_dict.keys())
    
    return outlier_dict, category_column

def remove_outliers_respiratory(df: pd.DataFrame, table_name: str='respiratory_support', 
                                file_path: str = '../config/outliers.json') -> pd.DataFrame:
    """
    Outlier removal for respiratory support data.
    Uses vectorized operations for better performance.
    """
    # Work on a copy to avoid modifying original
    df = df.copy()
    
    # Load outlier ranges
    range_dict, category_columns = read_outlier_file(table_name, file_path)
    
    if not range_dict:
        print("⚠️ No outlier ranges found. Returning original DataFrame.")
        return df
    
    # Process all columns at once using vectorized operations
    existing_columns = [col for col in category_columns if col in df.columns]
    
    if not existing_columns:
        print("⚠️ No matching columns found in DataFrame.")
        return df
    
    # Create arrays for min/max values for existing columns
    min_values = np.array([range_dict[col][0] for col in existing_columns])
    max_values = np.array([range_dict[col][1] for col in existing_columns])
    
    # Get subset of data for existing columns
    subset_data = df[existing_columns].values
    
    # Vectorized outlier detection
    outlier_mask = np.logical_and(
        ~np.isnan(subset_data),
        np.logical_or(subset_data < min_values, subset_data > max_values)
    )
    
    # Count outliers per column
    outlier_counts = np.sum(outlier_mask, axis=0)
    valid_counts = np.sum(~np.isnan(subset_data), axis=0)
    
    # Set outliers to NaN
    subset_data[outlier_mask] = np.nan
    df[existing_columns] = subset_data
    
    # Print summary
    total_outliers = np.sum(outlier_counts)
    if total_outliers > 0:
        print(f"Outliers found in {table_name}")
        for i, col in enumerate(existing_columns):
            if outlier_counts[i] > 0:
                perc_outlier = (outlier_counts[i] / valid_counts[i] * 100) if valid_counts[i] > 0 else 0
                print(f"⚠️ {outlier_counts[i]} rows ({perc_outlier:.2f}%) with outliers in '{col}' have been set to NaN.")
    else:
        print(f"✅ No outliers found in {table_name} table.")
    
    return df

def remove_outliers_general(df: pd.DataFrame, table_name: str, 
                            value_column: str, file_path: str = '../config/outliers.json') -> pd.DataFrame:
    
    # Load outlier ranges
    range_dict, category_column = read_outlier_file(table_name, file_path)
    
    if not range_dict:
        print("⚠️ No outlier ranges found. Returning original DataFrame.")
        return df
    
    # Check if required columns exist
    if category_column not in df.columns:
        print(f"⚠️ Category column '{category_column}' not found in DataFrame.")
        return df
    
    if value_column not in df.columns:
        print(f"⚠️ Value column '{value_column}' not found in DataFrame.")
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Get valid (non-null) rows for processing
    valid_mask = df[value_column].notna() & df[category_column].notna()
    
    if not valid_mask.any():
        print("⚠️ No valid data found for outlier detection.")
        return df
    
    # Work only with valid data to improve performance
    valid_data = df[valid_mask]
    
    # Vectorized threshold mapping
    categories = valid_data[category_column].values
    values = valid_data[value_column].values
    
    # Create threshold arrays
    min_thresholds = np.array([range_dict.get(cat, [0, float('inf')])[0] for cat in categories])
    max_thresholds = np.array([range_dict.get(cat, [0, float('inf')])[1] for cat in categories])
    
    # Vectorized outlier detection
    outlier_mask = (values < min_thresholds) | (values > max_thresholds)
    
    if outlier_mask.any():
        print(f"Outliers found in {table_name}")
        
        # Get indices of outliers in original dataframe
        outlier_indices = valid_data.index[outlier_mask]
        
        # Set outliers to NaN
        df.loc[outlier_indices, value_column] = np.nan
        
        # Count outliers by category for summary
        outlier_categories = valid_data.loc[outlier_mask, category_column]
        outlier_counts = outlier_categories.value_counts()
        
        # Print summary
        for category, count in outlier_counts.items():
            total = (df[category_column] == category).sum()
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"Category '{category}': {count} outliers ({percentage:.2f}%) have been set to NaN.")
    else:
        print(f"✅ No outliers found in {table_name} table.")
    
    return df

# Convenience function to choose the right method
def remove_outliers(df: pd.DataFrame, table_name: str, value_column: Optional[str] = None, 
                   file_path: str = '../config/outliers.json') -> pd.DataFrame:
    """
    Unified function to remove outliers based on table type.
    
    Args:
        df: Input DataFrame
        table_name: Type of table ('labs', 'vitals', 'respiratory_support')
        value_column: Column containing values (required for labs/vitals)
        file_path: Path to outlier configuration file
    
    Returns:
        DataFrame with outliers removed
    """
    if table_name == 'respiratory_support':
        return remove_outliers_respiratory(df, table_name, file_path)
    elif table_name in ['labs', 'vitals']:
        if value_column is None:
            raise ValueError(f"value_column is required for table_name '{table_name}'")
        return remove_outliers_general(df, table_name, value_column, file_path)
    else:
        raise ValueError(f"Unknown table_name: {table_name}")


def remove_outliers_with_timing(df: pd.DataFrame, table_name: str, 
                               value_column: Optional[str] = None, 
                               file_path: str = '../config/outliers.json') -> pd.DataFrame:
    """
    Remove outliers with performance monitoring.
    """
    import time
    
    start_time = time.time()
    result = remove_outliers(df, table_name, value_column, file_path)
    end_time = time.time()
    
    print(f"⏱️ Outlier removal completed in {end_time - start_time:.2f} seconds")
    print(f"🟢 Processed {len(df):,} rows")
    
    return result