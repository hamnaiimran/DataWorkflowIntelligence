import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

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
    if uploaded_file is None:
        st.warning("No file was uploaded.")
        return None
        
    try:
        data = pd.read_csv(uploaded_file)
        if data.empty:
            st.warning("The uploaded file is empty.")
            return None
        return data
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or contains no data.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure the file is properly formatted.")
        return None
    except UnicodeDecodeError:
        st.error("Error reading the file. Please ensure the file is encoded in UTF-8.")
        return None
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def fetch_yahoo_finance_data(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    symbol: str
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
    if not symbol:
        st.warning("No ticker symbol provided.")
        return None
        
    if not start_date or not end_date:
        st.warning("Please provide both start and end dates.")
        return None
        
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        st.warning("Start and end dates must be datetime objects.")
        return None
        
    if start_date > end_date:
        st.warning("Start date cannot be after end date.")
        return None
        
    if start_date > datetime.now():
        st.warning("Start date cannot be in the future.")
        return None
        
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Check if data is empty
        if data.empty:
            st.warning(f"No data found for ticker {symbol} in the specified date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).")
            return None
            
        return data
    except yf.errors.YFinanceError as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {str(e)}")
        return None

def get_summary_stats(df):
    """
    Get summary statistics of the dataframe
    
    Parameters:
    -----------
    df: pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of the dataframe
    """
    if df is None or df.empty:
        st.warning("No data provided for summary statistics.")
        return None
        
    try:
        return df.describe()
    except Exception as e:
        st.error(f"Error calculating summary statistics: {str(e)}")
        return None
