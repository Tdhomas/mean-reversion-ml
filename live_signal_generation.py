# live_signal_generation.py

import pandas as pd
import numpy as np
import ta
import joblib
import warnings
import datetime
import time
from ib_insync import IB, Stock, util
import pytz


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
        # Remove 'average' and 'barCount' columns if they exist
        df = df.drop(columns=['average', 'barCount'], errors='ignore')
        # Capitalize the first letter of each column name
        df.columns = [col.capitalize() for col in df.columns]
        # Update the DataFrame in the dictionary
        stock_data[symbol] = df
    return stock_data

# Function to load tickers from a CSV file
def load_tickers_from_csv(file_path):
    df = pd.read_csv(file_path)
    tickers = df['Symbol'].tolist()
    filtered_tickers = [ticker for ticker in tickers if '^' not in ticker and '/' not in ticker]
    return filtered_tickers

def feature_engineering(df):
    df = df.copy()
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
    
    #To remove for best_model_usable
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

def encode_date(df):
    df = df.copy()
    df['Date'] = df.index
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

def encode_symbol(df):
    df = df.copy()
    # Load LabelEncoder
    le = joblib.load(MODELS_DIR /'label_encoder.pkl')
    df['Symbol_Encoded'] = le.transform(df['Symbol'])
    return df

def data_cleaning(df):
    # Replace infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows where MA50 or MA200 are NaN
    df = df.dropna(subset=['MA50', 'MA200'])
    if df.empty:
        print("Dataframe is empty after dropping NaNs in MA50 and MA200.")
        return df
    # Impute missing values
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    # Create 'Close_Diff' as in training
    if 'Close' in df.columns:
        df['Close_Diff'] = df['Close'].diff()
        df['Close_Diff'].fillna(0, inplace=True)
    return df

def calculate_risk_management(df):
    df = df.copy()
    df['Entry_Price'] = df['Close']

    # For long positions (Predicted_Label == 1)
    df.loc[df['Predicted_Label'] == 1, 'Stop_Loss'] = df['Entry_Price'] - df['ATR'] * 2
    df.loc[df['Predicted_Label'] == 1, 'Take_Profit'] = df['Entry_Price'] + df['ATR'] * 3

    # For short positions (Predicted_Label == -1)
    df.loc[df['Predicted_Label'] == -1, 'Stop_Loss'] = df['Entry_Price'] + df['ATR'] * 2
    df.loc[df['Predicted_Label'] == -1, 'Take_Profit'] = df['Entry_Price'] - df['ATR'] * 3

    return df

def calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss_price):
    # Check if entry_price or stop_loss_price is NaN
    if pd.isna(entry_price) or pd.isna(stop_loss_price):
        return 0  # or you can return np.nan if you prefer not to have 0 as a placeholder

    # risk_per_trade is a fraction of portfolio value (e.g., 0.01 for 1%)
    # Compute risk amount in dollars
    risk_amount = portfolio_value * risk_per_trade

    # Price difference between entry and stop-loss
    price_difference = abs(entry_price - stop_loss_price)

    if price_difference == 0:
        return 0  # Avoid division by zero

    # Number of shares = risk amount / price difference
    position_size = risk_amount / price_difference

    if position_size > 10:
        position_size = 10

    return int(position_size)

if __name__ == "__main__":
    # Load stock symbols
    nasdaq_tickers = load_tickers_from_csv(DATA_DIR /'trading_stocks.csv')
    print(f"Total tickers loaded: {len(nasdaq_tickers)}")

    # Define parameters for historical data
    duration = '3 M'           # Adjust as needed (e.g., '1 Y' for one year)
    bar_size = '1 hour'        # '1 hour' for hourly data
    what_to_show = 'TRADES'    # 'TRADES' to include Volume
    
    # Set timezone to US/Eastern
    eastern = pytz.timezone('US/Eastern')
    # Get current time in US/Eastern timezone
    current_time = datetime.datetime.now(eastern)

    # Format the date and time as 'YYYYMMDD HH:MM:SS Timezone'
    end_date = current_time.strftime('%Y%m%d %H:%M:%S US/Eastern')
    # end_date = "20241001 23:59:00 US/Eastern"
    

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

    all_signals = []
    
    # Load wokring trained model
    model = joblib.load(MODELS_DIR / 'best_model.pkl')

    # Assume a portfolio value
    portfolio_value = 100000  # Adjust as needed
    risk_per_trade = 0.01     # 1% risk per trade

    # Specify the number of recent days to print signals for
    recent_days = 2  # Modify this value as needed

    for symbol, df in stock_data.items():
        # Feature Engineering
        df = feature_engineering(df)
        if df.empty:
            continue
        # Encode Symbol
        df = encode_symbol(df)
        # Encode Date
        df = encode_date(df)
        # Data Cleaning
        df = data_cleaning(df)
        if df.empty:
            continue
        # Prepare features
        features = df.select_dtypes(include=[np.number])
        # Predict
        df['Predicted_Label'] = model.predict(features)
        # Calculate risk management parameters
        df = calculate_risk_management(df)
        # Calculate position size
        df['Position_Size'] = df.apply(lambda row: calculate_position_size(
            portfolio_value, risk_per_trade, row['Entry_Price'], row['Stop_Loss']), axis=1)
        # Estimate time horizon (you can adjust this logic)
        df['Time_Horizon'] = 'Short-term'
        # Collect signals
        signals = df[df['Predicted_Label'] != 0].copy()
        if not signals.empty:
            signals['Action'] = signals['Predicted_Label'].apply(lambda x: 'Buy' if x == 1 else 'Sell Short')
            signals['Symbol'] = symbol
            all_signals.append(signals)

    # Combine all signals
    if all_signals:
        all_signals_df = pd.concat(all_signals)
        # Save all signals to CSV
        all_signals_df.to_csv(DATA_DIR / 'live_signals_all.csv', index=False)
        print("All signals saved to 'live_signals_all.csv'.")
        
        # Set timezone to US/Eastern
        eastern = pytz.timezone('US/Eastern')

        # Make cutoff_date timezone-aware
        cutoff_date = eastern.localize(datetime.datetime.now()) - datetime.timedelta(days=recent_days)

        # Ensure 'Date' column in DataFrame is also timezone-aware (if it isn't already)
        #all_signals_df['Date'] = pd.to_datetime(all_signals_df['Date']).dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')

        recent_signals = all_signals_df[all_signals_df['Date'] >= cutoff_date]

        if not recent_signals.empty:
            print(f"Generated Signals in the last {recent_days} days:")
            recent_signals['Date'] = pd.to_datetime(recent_signals['Date'])
            # Sort the DataFrame by the 'Date' column in ascending order
            main_data = recent_signals[['Date', 'Symbol', 'Action', 'Entry_Price', 'Stop_Loss', 'Take_Profit', 'Position_Size', 'Time_Horizon']].sort_values(by='Date')
            print(main_data)
            # Optionally, save recent signals to a separate CSV
            main_data.to_csv(DATA_DIR / 'live_signals_recent.csv', index=False)
            print("Data saved to live_signals_recent.csv")
        else:
            print(f"No signals generated in the last {recent_days} days.")
    else:
        print("No signals generated.")