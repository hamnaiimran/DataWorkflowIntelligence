import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def check_missing_values(data):
    """
    Check for missing values in the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
        
    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing_values = data.isnull().sum()
    missing_percent = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage (%)': missing_percent
    })
    # Only return columns with missing values
    return missing_df[missing_df['Missing Values'] > 0]

def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the dataframe"""
    try:
        # Create imputer
        imputer = SimpleImputer(strategy=strategy)
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Impute missing values
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        return df
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df

def remove_outliers(data, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        Input data
    columns: list
        List of column names to check for outliers
    method: str
        Method to use for outlier detection ('iqr' or 'zscore')
    threshold: float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Data with outliers removed
    """
    # Make a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    for col in columns:
        if col not in processed_data.columns or not pd.api.types.is_numeric_dtype(processed_data[col]):
            continue
            
        if method == 'iqr':
            # IQR method
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            processed_data = processed_data[(processed_data[col] >= lower_bound) & 
                                          (processed_data[col] <= upper_bound)]
        elif method == 'zscore':
            # Z-score method
            mean = processed_data[col].mean()
            std = processed_data[col].std()
            z_scores = abs((processed_data[col] - mean) / std)
            processed_data = processed_data[z_scores <= threshold]
    
    return processed_data

def scale_features(df, method='standard'):
    """Scale features using specified method"""
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()
        
        # Scale features
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    except Exception as e:
        st.error(f"Error scaling features: {str(e)}")
        return df

def encode_categorical(df):
    """Encode categorical variables"""
    try:
        # Get categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, columns=categorical_cols)
        
        return df
    except Exception as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        return df
