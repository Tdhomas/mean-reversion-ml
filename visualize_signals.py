# visualize_signals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

def plot_signals(df, symbol, output_dir='plots'):
    """
    Plots the stock price with buy/sell signals and technical indicators.

    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data with signals and indicators.
    - symbol (str): Stock symbol.
    - output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the plot style
    sns.set(style="darkgrid")
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

    # 1. Price with Bollinger Bands and Buy/Sell Signals
    axs[0].plot(df.index, df['Close'], label='Close Price', color='blue')
    axs[0].fill_between(df.index, df['Bollinger_High'], df['Bollinger_Low'], color='grey', alpha=0.1, label='Bollinger Bands')

    # Plot buy signals
    buy_signals = df[df['Label'] == 1]
    axs[0].scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)

    # Plot sell signals
    sell_signals = df[df['Label'] == -1]
    axs[0].scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)

    axs[0].set_title(f'{symbol} Price with Buy/Sell Signals and Bollinger Bands')
    axs[0].set_ylabel('Price')
    axs[0].legend()

    # 2. RSI
    axs[1].plot(df.index, df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, color='red', linestyle='--', label='Overbought (70)')
    axs[1].axhline(30, color='green', linestyle='--', label='Oversold (30)')
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].set_ylabel('RSI')
    axs[1].legend()

    # 3. Stochastic Oscillator
    axs[2].plot(df.index, df['Stoch_%K'], label='%K', color='blue')
    axs[2].plot(df.index, df['Stoch_%D'], label='%D', color='orange')
    axs[2].axhline(80, color='red', linestyle='--', label='Overbought (80)')
    axs[2].axhline(20, color='green', linestyle='--', label='Oversold (20)')
    axs[2].set_title('Stochastic Oscillator')
    axs[2].set_ylabel('Stochastic')
    axs[2].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"{symbol}_signals.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved for {symbol} at {plot_filename}")

# Optional: Interactive Plot with Plotly
def plot_signals_interactive(df, symbol, output_dir='plots_interactive'):
    """
    Creates an interactive plot using Plotly.

    Parameters:
    - df (pd.DataFrame): DataFrame containing stock data with signals and indicators.
    - symbol (str): Stock symbol.
    - output_dir (str): Directory to save the plots.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=(f'{symbol} Price with Buy/Sell Signals',
                                        'Relative Strength Index (RSI)',
                                        'Stochastic Oscillator'),
                        vertical_spacing=0.05)

    # Price and Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_High'], mode='lines', name='Bollinger High', line=dict(color='grey', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Low'], mode='lines', name='Bollinger Low', line=dict(color='grey', dash='dash')), row=1, col=1)

    # Define buy_signals and sell_signals
    buy_signals = df[df['Label'] == 1]
    sell_signals = df[df['Label'] == -1]

    # Buy signals
    fig.add_trace(go.Scatter(mode='markers', name='Buy Signal',
                             x=buy_signals.index, y=buy_signals['Close'],
                             marker=dict(symbol='triangle-up', color='green', size=12)),
                  row=1, col=1)
    
    # Sell signals
    fig.add_trace(go.Scatter(mode='markers', name='Sell Signal',
                             x=sell_signals.index, y=sell_signals['Close'],
                             marker=dict(symbol='triangle-down', color='red', size=12)),
                  row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')), row=2, col=1)

    # Stochastic Oscillator
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_%K'], mode='lines', name='%K'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_%D'], mode='lines', name='%D'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[80]*len(df), mode='lines', name='Overbought (80)', line=dict(color='red', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[20]*len(df), mode='lines', name='Oversold (20)', line=dict(color='green', dash='dash')), row=3, col=1)

    fig.update_layout(height=1400, width=1200, title_text=f'{symbol} Technical Indicators and Signals')
    plot_filename = os.path.join(output_dir, f"{symbol}_signals_interactive.html")
    fig.write_html(plot_filename)
    print(f"Interactive plot saved for {symbol} at {plot_filename}")

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize Buy/Sell Signals and Technical Indicators for Stocks.')
    parser.add_argument('--data', type=str, default=DATA_DIR / 'filtered_dataset.csv', help='Path to the processed dataset CSV file.')
    parser.add_argument('--symbol', type=str, help='Stock symbol to visualize. If not provided, all symbols will be visualized.')
    parser.add_argument('--interactive', action='store_true', help='Generate interactive plots using Plotly.')
    args = parser.parse_args()

    # Load the dataset
    if not os.path.exists(args.data):
        print(f"Data file {args.data} does not exist. Please ensure the path is correct.")
        exit()
    
    dataset = pd.read_csv(args.data, parse_dates=['Date'])
    dataset.set_index('Date', inplace=True)

    # Get unique symbols
    symbols = dataset['Symbol'].unique()
    
    # Determine which symbols to plot
    if args.symbol:
        symbols_to_plot = [args.symbol] if args.symbol in symbols else []
        if not symbols_to_plot:
            print(f"Symbol {args.symbol} not found in the dataset.")
            exit()
    else:
        symbols_to_plot = list(symbols)  # Convert to list for consistent behavior

    if not symbols_to_plot:
        print("No symbols to plot.")
        exit()

    for symbol in symbols_to_plot:
        df_symbol = dataset[dataset['Symbol'] == symbol].copy()
        df_symbol.sort_index(inplace=True)  # Ensure data is sorted by date

        if df_symbol.empty:
            print(f"No data available for {symbol}. Skipping.")
            continue

        if args.interactive:
            plot_signals_interactive(df_symbol, symbol)
        else:
            plot_signals(df_symbol, symbol)