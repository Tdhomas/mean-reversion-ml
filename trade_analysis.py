# trade_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# For machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings
import warnings
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

def load_trades_data(trades_file= DATA_DIR / 'backtest_trades.csv'):
    if os.path.exists(trades_file):
        trades_df = pd.read_csv(trades_file)
        trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
        trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
        return trades_df
    else:
        print(f"{trades_file} not found.")
        return None

def analyze_trades(trades_df):
    # Separate winning and losing trades
    winning_trades = trades_df[trades_df['Outcome'] == 'Win']
    losing_trades = trades_df[trades_df['Outcome'] == 'Loss']

    # Descriptive statistics
    win_features = winning_trades.describe()
    loss_features = losing_trades.describe()
    print("Winning Trades Statistics:")
    print(win_features)
    print("\nLosing Trades Statistics:")
    print(loss_features)

    # Correlation matrix (excluding non-numeric columns)
    print("\nCorrelation Matrix:")
    numeric_cols = trades_df.select_dtypes(include=[np.number])  # Select only numeric columns
    correlation = numeric_cols.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(correlation, annot=False, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Trade Features')
    plt.show()

def feature_importance_analysis(trades_df):
    # Prepare data
    features = ['RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low',
                'Stoch_%K', 'Stoch_%D', 'Z-Score', 'Williams_%R',
                'ATR', 'MA50', 'MA200', 'Price_Momentum',
                'Volume_Change']  # Add any other relevant features

    # Ensure all features are present in the DataFrame
    available_features = [feature for feature in features if feature in trades_df.columns]

    X = trades_df[available_features]
    y = trades_df['Outcome'].map({'Win': 1, 'Loss': 0})

    # Handle any missing values
    X = X.fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=available_features)
    importances = importances.sort_values(ascending=True)
    importances.plot(kind='barh', figsize=(10,8))
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()

def plot_trades(trades_df, signals_df, symbol):
    import matplotlib.dates as mdates

    # Filter data for the specific symbol
    symbol_data = signals_df[signals_df['Symbol'] == symbol].copy()
    symbol_data['Date'] = pd.to_datetime(symbol_data['Date'])
    symbol_data.set_index('Date', inplace=True)
    symbol_data.sort_index(inplace=True)
    symbol_data['Close'] = symbol_data['Close'].astype(float)

    # Plot price
    plt.figure(figsize=(14,7))
    plt.plot(symbol_data.index, symbol_data['Close'], label='Close Price', color='blue')

    # Plot trades
    symbol_trades = trades_df[trades_df['Symbol'] == symbol]
    for idx, trade in symbol_trades.iterrows():
        entry_date = trade['Entry Date']
        exit_date = trade['Exit Date']
        entry_price = trade['Entry Price']
        exit_price = trade['Exit Price']
        outcome = trade['Outcome']
        if outcome == 'Win':
            color = 'green'
        else:
            color = 'red'
        plt.scatter(entry_date, entry_price, marker='^', color=color, s=100, label='Entry' if idx==0 else "")
        plt.scatter(exit_date, exit_price, marker='v', color=color, s=100, label='Exit' if idx==0 else "")
        plt.plot([entry_date, exit_date], [entry_price, exit_price], color=color, linestyle='--')

    plt.title(f'Trades for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Load trades data
    trades_df = load_trades_data(DATA_DIR / 'backtest_trades.csv')
    if trades_df is None:
        return

    # Analyze trades
    analyze_trades(trades_df)

    # Feature importance analysis
    feature_importance_analysis(trades_df)

    # Load signals data
    if os.path.exists(DATA_DIR / 'buy_signals.csv') and os.path.exists(DATA_DIR / 'sell_signals.csv'):
        buy_signals = pd.read_csv(DATA_DIR / 'buy_signals.csv')
        sell_signals = pd.read_csv(DATA_DIR / 'sell_signals.csv')
        signals_df = pd.concat([buy_signals, sell_signals], ignore_index=True)
        # Ensure date column is in datetime format
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    else:
        print("Buy and sell signals files not found.")
        return

    # Plot trades for a specific symbol
    symbol = 'IPAR'  # Replace with the symbol you want to analyze
    plot_trades(trades_df, signals_df, symbol)

if __name__ == "__main__":
    main()