#data_preparation.py

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from ib_insync import IB, Stock, util
import datetime
import time
import joblib
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

from pathlib import Path

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
PARAMS_DIR = Path("params")
PLOTS_DIR = Path("plots")

# Create output dirs on demand when the script writes files
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Function to load tickers from a CSV file
def load_tickers_from_csv(file_path):
    df = pd.read_csv(file_path)
    tickers = df['Symbol'].tolist()
    filtered_tickers = [ticker for ticker in tickers if '^' not in ticker and '/' not in ticker]
    return filtered_tickers

# Function to download stock data using yfinance
def download_stock_data(tickers, start_date, end_date):
    stock_data = {}
    for symbol in tickers:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            if not data.empty:
                if data.isnull().sum().sum() > 0:
                    print(f"Data for {symbol} contains missing values.")
                stock_data[symbol] = data
                print(f"Downloaded data for {symbol}")
            else:
                print(f"No data for {symbol}")
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
    return stock_data

# Function to save data to disk
def save_data(stock_data, filename= DATA_DIR / 'stock_data.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(stock_data, f)
    print(f"Stock data saved to {filename}.")

# Function to load data from disk
def load_saved_data(filename= DATA_DIR / 'stock_data.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            stock_data = pickle.load(f)
        print(f"Stock data loaded from {filename}.")
        return stock_data
    else:
        print(f"No saved data found at {filename}.")
        return None


# Function to download stock data using ib_insync
def download_stock_data_ib(tickers, end_date, duration, bar_size, what_to_show, use_rth=True, timeout=10):
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=1)  # Adjust port and clientId as needed
        print("Connected to Interactive Brokers.")
    except Exception as e:
        print(f"Failed to connect to IB: {e}")
        return {}

    stock_data = {}
    for symbol in tickers:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            # Request historical data
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,       # e.g., '3 M' for three months
                barSizeSetting=bar_size,    # e.g., '1 hour'
                whatToShow=what_to_show,     # e.g., 'TRADES'
                useRTH=use_rth,
                formatDate=1,
                timeout=timeout
            )

            if bars:
                df = util.df(bars)
                df['Datetime'] = df['date']
                df.set_index('Datetime', inplace=True)
                df.drop(columns=['date'], inplace=True)
                df['Symbol'] = symbol
                stock_data[symbol] = df
                print(f"Downloaded data for {symbol}")
            else:
                print(f"No data returned for {symbol}")

            # Respect IB rate limits
            time.sleep(1)  # Adjust sleep time as necessary

        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")

    ib.disconnect()
    print("Disconnected from Interactive Brokers.")
    return stock_data


# Function to format stock data
def format_and_combine_data(stock_data):
    for symbol, df in stock_data.items():
        df['Symbol'] = symbol
        # Remove 'average' and 'barCount' columns if they exist
        df = df.drop(columns=['average', 'barCount'], errors='ignore')
        # Capitalize the first letter of each column name
        df.columns = [col.capitalize() for col in df.columns]
        # Update the DataFrame in the dictionary
        stock_data[symbol] = df
    return stock_data


# Function to calculate technical features
def feature_engineering(df):
    print(f"Columns in DataFrame: {df.columns.tolist()}")
    # Moving Averages
    df['MA50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
    df['MA200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
    
    # Average True Range
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Relative Strength Index
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # Price Momentum
    df['Price_Momentum'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
    
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    # Keltner Channels
    keltner = ta.volatility.KeltnerChannel(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    df['Keltner_High'] = keltner.keltner_channel_hband()
    df['Keltner_Low'] = keltner.keltner_channel_lband()
    df['Keltner_Middle'] = keltner.keltner_channel_mband()
    
    # Z-Score
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    df['Z-Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # Williams %R
    williams_r = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
    df['Williams_%R'] = williams_r.williams_r()
    
    # Drop NaN values
    df.dropna(inplace=True)
    return df

def generate_signals(df, params):
    df = df.copy()
    df['Signal'] = 0

    # Get parameters, use defaults if not provided
    rsi_buy = params.get('rsi_buy', 30)
    rsi_sell = params.get('rsi_sell', 70)
    stoch_k_buy = params.get('stoch_k_buy', 20)
    stoch_k_sell = params.get('stoch_k_sell', 80)
    bollinger_low_multiplier = params.get('bollinger_low_multiplier', 1.0)
    bollinger_high_multiplier = params.get('bollinger_high_multiplier', 1.0)
    zscore_buy = params.get('zscore_buy', -2)
    zscore_sell = params.get('zscore_sell', 2)
    williams_r_buy = params.get('williams_r_buy', -80)
    williams_r_sell = params.get('williams_r_sell', -20)

    in_buy_signal = False
    in_sell_signal = False

    for i in range(1, len(df)):
        # Buy signal conditions
        rsi_buy_condition = df['RSI'].iloc[i] <= rsi_buy
        stoch_buy_condition = df['Stoch_%K'].iloc[i] <= stoch_k_buy and df['Stoch_%K'].iloc[i] > df['Stoch_%D'].iloc[i]
        bollinger_buy_condition = df['Close'].iloc[i] <= df['Bollinger_Low'].iloc[i] * bollinger_low_multiplier
        keltner_buy_condition = df['Close'].iloc[i] <= df['Keltner_Low'].iloc[i]
        zscore_buy_condition = df['Z-Score'].iloc[i] <= zscore_buy
        williams_r_buy_condition = df['Williams_%R'].iloc[i] <= williams_r_buy

        # Strategy conditions
        
        # #Strategy 1 :
        # strategy_1 = rsi_buy_condition and bollinger_buy_condition
        # strategy_2 = rsi_buy_condition and stoch_buy_condition
        # strategy_3 = bollinger_buy_condition and stoch_buy_condition
        
        # Strategy 2 :
        # strategy_1 = rsi_buy_condition and bollinger_buy_condition
        # strategy_2 = rsi_buy_condition and stoch_buy_condition
        # strategy_3 = bollinger_buy_condition and stoch_buy_condition
        # strategy_4 = rsi_buy_condition and williams_r_buy_condition
        # strategy_5 = rsi_buy_condition and zscore_buy_condition
        # strategy_6 = rsi_buy_condition and keltner_buy_condition
        

        # Test Strategy
        strategy_1 = stoch_buy_condition and bollinger_buy_condition and zscore_buy_condition
        strategy_2 = stoch_buy_condition and bollinger_buy_condition and keltner_buy_condition and williams_r_buy_condition
        strategy_3 = rsi_buy_condition and stoch_buy_condition and bollinger_buy_condition

        
        # Combine all buy strategies
        buy_conditions = [
            strategy_1, strategy_2, strategy_3
        ]
        # , strategy_7, strategy_8, strategy_9, strategy_10

        if not in_buy_signal and any(buy_conditions):
            df.at[df.index[i], 'Signal'] = 1
            in_buy_signal = True
            in_sell_signal = False  # Reset sell signal state
        else:
            # Check if we have exited the buy condition
            if not any([rsi_buy_condition, stoch_buy_condition, bollinger_buy_condition, keltner_buy_condition, zscore_buy_condition, williams_r_buy_condition]):
                in_buy_signal = False

        # Sell signal conditions
        rsi_sell_condition = df['RSI'].iloc[i] >= rsi_sell
        stoch_sell_condition = df['Stoch_%K'].iloc[i] >= stoch_k_sell and df['Stoch_%K'].iloc[i] < df['Stoch_%D'].iloc[i]
        bollinger_sell_condition = df['Close'].iloc[i] >= df['Bollinger_High'].iloc[i] * bollinger_high_multiplier
        keltner_sell_condition = df['Close'].iloc[i] >= df['Keltner_High'].iloc[i]
        zscore_sell_condition = df['Z-Score'].iloc[i] >= zscore_sell
        williams_r_sell_condition = df['Williams_%R'].iloc[i] >= williams_r_sell

        # Combine sell conditions using same structure as buy conditions
        
        # #Strategy 1 :
        # strategy_1b = rsi_sell_condition and bollinger_sell_condition
        # strategy_2b = rsi_sell_condition and stoch_sell_condition
        # strategy_3b = bollinger_sell_condition and stoch_sell_condition
        
        # Strategy 2 :
        # strategy_1b = rsi_sell_condition and bollinger_sell_condition
        # strategy_2b = rsi_sell_condition and stoch_sell_condition
        # strategy_3b = bollinger_sell_condition and stoch_sell_condition
        # strategy_4b = rsi_sell_condition and williams_r_sell_condition
        # strategy_5b = rsi_sell_condition and zscore_sell_condition
        # strategy_6b = rsi_sell_condition and keltner_sell_condition
        
        # Test Strategy
        strategy_1b = stoch_sell_condition and bollinger_sell_condition and zscore_sell_condition
        strategy_2b = stoch_sell_condition and bollinger_sell_condition and keltner_sell_condition and williams_r_sell_condition
        strategy_3b = rsi_sell_condition and bollinger_sell_condition and stoch_sell_condition

        
        sell_conditions = [
            strategy_1b, strategy_2b, strategy_3b
        ]
        # , strategy_7b, strategy_8b, strategy_9b, strategy_10b


        if not in_sell_signal and any(sell_conditions):
            df.at[df.index[i], 'Signal'] = -1
            in_sell_signal = True
            in_buy_signal = False  # Reset buy signal state
        else:
            # Check if we have exited the sell condition
            if not any([rsi_sell_condition, stoch_sell_condition, bollinger_sell_condition, keltner_sell_condition, zscore_sell_condition, williams_r_sell_condition]):
                in_sell_signal = False

    return df

# Function to evaluate features: daily mean, percent change, etc.
def evaluate_features(df, feature_list=None):
    if feature_list is None:
        feature_list = ['Close', 'Volume', 'MA50', 'MA200', 'ATR', 'RSI', 'Price_Momentum', 'Volume_Change']

    evaluation_results = {}

    for feature in feature_list:
        if feature in df.columns:
            # Daily mean
            daily_mean = df[feature].mean()
            
            # Daily percent change mean
            percent_change_mean = df[feature].pct_change().mean()

            # Rolling statistics over different time frames (7 days, 30 days, 90 days)
            rolling_mean_7 = df[feature].rolling(window=7).mean().mean()
            rolling_mean_30 = df[feature].rolling(window=30).mean().mean()
            rolling_mean_90 = df[feature].rolling(window=90).mean().mean()
            
            evaluation_results[feature] = {
                'daily_mean': daily_mean,
                'percent_change_mean': percent_change_mean,
                'rolling_mean_7': rolling_mean_7,
                'rolling_mean_30': rolling_mean_30,
                'rolling_mean_90': rolling_mean_90
            }

    return pd.DataFrame(evaluation_results)

# Function to encode date into cyclical features
def encode_date(df):
    df = df.copy()
    df['Date'] = df.index  # Ensure 'Date' is a column
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    # Cyclical encoding
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    # Drop unnecessary columns
    df = df.drop(['Day', 'Month', 'Year'], axis=1)
    return df

# Function to encode stock symbols into numeric format
def encode_symbol(df, le=None):
    df = df.copy()
    if le is None:
        # Create a new LabelEncoder and fit it
        le = LabelEncoder()
        le.fit(df['Symbol'].unique())
        joblib.dump(le,  MODELS_DIR / 'label_encoder.pkl')
        print("New LabelEncoder created and saved to 'label_encoder.pkl'.")
    elif isinstance(le, str):
        # Load LabelEncoder from the provided filename
        le = joblib.load(le)
        print(f"LabelEncoder loaded from '{le}'.")
    elif isinstance(le, LabelEncoder):
        # Use the provided LabelEncoder instance
        print("Using the provided LabelEncoder instance.")
    else:
        raise ValueError("The 'le' parameter must be None, a filename string, or a LabelEncoder instance.")
    
    # Transform the 'Symbol' column
    df['Symbol_Encoded'] = le.transform(df['Symbol'])
    return df


# Function to clean and preprocess data
def data_cleaning(df):
    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows where MA50 or MA200 are NaN
    df = df.dropna(subset=['MA50', 'MA200'])
    if df.empty:
        print("Dataframe is empty after dropping NaNs in MA50 and MA200.")
        return df
    # Handle outliers using the IQR method
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    multiplier = 3
    is_outlier = ((df[numeric_columns] < (Q1 - multiplier * IQR)) | (df[numeric_columns] > (Q3 + multiplier * IQR))).any(axis=1)
    df = df[~is_outlier]
    if df.empty:
        print("Dataframe is empty after outlier removal.")
        return df
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    imputed_data = imputer.fit_transform(df[numeric_columns])
    df_imputed = pd.DataFrame(imputed_data, columns=numeric_columns, index=df.index)
    df[numeric_columns] = df_imputed
    # Make data stationary using differencing
    if 'Close' in df.columns:
        df['Close_Diff'] = df['Close'].diff()
        df['Close_Diff'].fillna(0, inplace=True)
    return df

# Function to label data based on the combined strategy
def label_data(df, params):
    df = df.copy()
    df = generate_signals(df, params)
    # Rename 'Signal' to 'Label' or keep it as 'Signal' for consistency
    df.rename(columns={'Signal': 'Label'}, inplace=True)
    return df

# Function to log dataframe details during different stages
def log_dataframe_info(df, stage):
    print(f"Data after {stage}:")
    print(f"Shape: {df.shape}")
    print(df.head())

# Function to filter stocks based on the combined strategy's performance
def filter_optimal_stocks(combined_df, threshold=0.3):
    """
    Filters stocks based on the frequency and reliability of the combined strategy signals.

    Parameters:
    - combined_df (pd.DataFrame): The combined dataset for all stocks.
    - threshold (float): The minimum proportion of time a stock must meet the combined strategy criteria to be considered optimal.

    Returns:
    - optimal_symbols (list): List of stock symbols that consistently align with the combined strategy.
    """
    # Group by Symbol
    grouped = combined_df.groupby('Symbol')
    
    optimal_symbols = set()

    for symbol, group in grouped:
        # Ensure sufficient data points for analysis
        if len(group) < 50:  # Minimum data points to avoid filtering on insufficient data
            continue

        # Combined Strategy Conditions
        combined_conditions = (
            (group['Combined_Strategy'] == 1) 
        )
        
        # Calculate the proportion of time the combined conditions are met
        combined_proportion = combined_conditions.mean()

        # If the stock meets the combined strategy conditions more than the threshold, add it to the list
        if combined_proportion >= threshold:
            optimal_symbols.add(symbol)

    print(f"Number of optimal symbols after filtering: {len(optimal_symbols)}")
    return list(optimal_symbols)


# Function to prepare the dataset
def prepare_dataset(stock_data, threshold=0.3):
    all_data = []
    le = LabelEncoder()
    symbols = list(stock_data.keys())
    le.fit(symbols)
    # Save the LabelEncoder
    joblib.dump(le,  MODELS_DIR / 'label_encoder.pkl')
    print("LabelEncoder fitted and saved to 'label_encoder.pkl'.")

    for symbol, data in stock_data.items():
        if len(data) < 200:
            print(f"Not enough data for {symbol} to compute MA200.")
            continue
        data['Symbol'] = symbol  # Keep 'Symbol' for encoding and later use
        # Load parameters for the current stock
        if symbol in best_parameters_per_stock:
            params = best_parameters_per_stock[symbol]
        else:
            params = {}  # Use default parameters
            print(f"No parameters found for {symbol}, using default parameters.")
        data = feature_engineering(data)
        if data.empty:
            print(f"No data for {symbol} after feature engineering.")
            continue
        data = encode_symbol(data, le=le)  # Pass the LabelEncoder instance
        data = encode_date(data)
        data = data_cleaning(data)
        if data.empty:
            print(f"No data for {symbol} after cleaning.")
            continue
        data = label_data(data, params)
        if data.empty:
            print(f"No data for {symbol} after labeling.")
            continue
        log_dataframe_info(data, f"processing {symbol}")
        all_data.append(data)

    if not all_data:
        print("No data available after processing.")
        return pd.DataFrame()

    combined_data = pd.concat(all_data, ignore_index=True)

    # Evaluate features
    print("Evaluating features for the dataset...")
    feature_eval_df = evaluate_features(combined_data)
    print(feature_eval_df)

    # Return the prepared dataset
    return combined_data


if __name__ == "__main__":
    # Toggle to decide whether to download data or load saved data
    download_new_data = True  # Set to True to download new data, False to load existing data
    hourly_data = True  # Set to True to download hourly data, False to load daily data

    processed_data_filename = DATA_DIR / 'filtered_dataset.csv'
    label_encoder_filename = MODELS_DIR / 'label_encoder.pkl'
    
    if hourly_data:
        data_filename = DATA_DIR / 'stock_data_ib.pkl'
    else:
        data_filename = DATA_DIR / 'stock_data.pkl'

    if download_new_data:
        # Load stock symbols
        nasdaq_tickers = load_tickers_from_csv(DATA_DIR / "trading_stocks.csv")
        print(f"Total tickers loaded: {len(nasdaq_tickers)}")
        
        if hourly_data:
            # Define parameters for historical data
            duration = '3 M'           # Adjust as needed (e.g., '1 Y' for one year)
            bar_size = '1 hour'        # '1 hour' for hourly data
            what_to_show = 'TRADES'     # Changed from 'MIDPOINT' to 'TRADES' to include Volume
            end_date = "20240930 23:59:00 US/Eastern"
    
            # Download stock data using ib_insync
            stock_data = download_stock_data_ib(
                tickers=nasdaq_tickers,
                end_date=end_date,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show
                )
            
            # Format and combine data
            stock_data = format_and_combine_data(stock_data)
        
            # Save downloaded data
            save_data(stock_data, filename=data_filename)
            
        # Daily
        else: 
            # Define date range
            start_date = '2020-01-01'
            end_date = '2024-09-20'
        
            # Download stock data
            stock_data = download_stock_data(nasdaq_tickers, start_date, end_date)
        
            # Save downloaded data
            save_data(stock_data, filename=data_filename)
    else:
        # Load saved data
        stock_data = load_saved_data(filename=data_filename)
        
        if hourly_data:
            # Format and combine data
            stock_data = format_and_combine_data(stock_data)

        if stock_data is None:
            print("No data loaded. Exiting.")
            exit()

    # Load the best parameters per stock
    with open(PARAMS_DIR / "best_parameters_per_stock_indicators.pkl", 'rb') as f:
        best_parameters_per_stock = pickle.load(f)

    # Prepare dataset
    dataset = prepare_dataset(stock_data, threshold=0.05)
    
    # Save dataset to CSV
    if not dataset.empty:
        dataset.to_csv(processed_data_filename, index=False)
        print(f"Filtered dataset saved successfully to {processed_data_filename}.")
    else:
        print("No data to save after processing.")