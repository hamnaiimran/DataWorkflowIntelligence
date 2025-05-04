import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import datetime

def load_csv_data(uploaded_file):
    """
    Load data from an uploaded CSV file
    
    Parameters:
    -----------
    uploaded_file: StreamlitUploadedFile
        The uploaded CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the loaded data
    """
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def fetch_yahoo_finance_data(ticker, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker: str
        Stock ticker symbol
    start_date: datetime
        Start date for fetching data
    end_date: datetime
        End date for fetching data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Check if data is empty
        if data.empty:
            st.error(f"No data found for ticker {ticker} in the specified date range")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return None

def get_summary_stats(data):
    """
    Generate summary statistics for the data
    
    Parameters:
    -----------
    data: pd.DataFrame
        The input data
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of the data
    """
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        summary = numeric_data.describe()
        return summary
    except Exception as e:
        st.error(f"Error generating summary statistics: {e}")
        return None
