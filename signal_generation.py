# signal_generation.py

import pandas as pd
import joblib
import numpy as np

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

# def calculate_risk_management(df):
#     df = df.copy()
#     df['Entry_Price'] = df['Close']
#     # Stop-Loss: Entry Price minus 1.5 * ATR
#     df['Stop_Loss'] = df['Entry_Price'] - df['ATR'] * 1.5
#     # Take-Profit: Entry Price plus 3 * ATR
#     df['Take_Profit'] = df['Entry_Price'] + df['ATR'] * 1.5
#     return df

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

if __name__ == "__main__":
    # Load the dataset
    dataset = pd.read_csv(DATA_DIR / 'filtered_dataset.csv')

    # Load the trained model
    model = joblib.load(MODELS_DIR / 'best_model.pkl')

    # Prepare features for prediction
    features = dataset.drop(columns=['Label'], errors='ignore')
    features = features.select_dtypes(include=[np.number])

    # Predict using the model
    dataset['Predicted_Label'] = model.predict(features)

    # Filter buy and sell signals
    buy_signals = dataset[dataset['Predicted_Label'] == 1].copy()
    sell_signals = dataset[dataset['Predicted_Label'] == -1].copy()

    # Apply risk management
    buy_signals = calculate_risk_management(buy_signals)
    sell_signals = calculate_risk_management(sell_signals)

    # Save signals
    buy_signals.to_csv(DATA_DIR / 'buy_signals.csv', index=False)
    sell_signals.to_csv(DATA_DIR / 'sell_signals.csv', index=False)
