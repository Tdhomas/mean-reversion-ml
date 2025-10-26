# backtesting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pickle
import os
import math

# For date handling
from datetime import datetime, timedelta

# For performance metrics
from scipy.stats import norm

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


def annualize_rets(r, periods_per_year=252):
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year / n_periods) - 1

def annualize_vol(r, periods_per_year=252):
    return r.std() * np.sqrt(periods_per_year)

def sharpe_ratio_func(r, riskfree_rate=0.03, periods_per_year=252):
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def sortino_ratio_func(r, riskfree_rate=0.03, periods_per_year=252):
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    downside_ret = excess_ret[excess_ret < 0]
    expected_return = excess_ret.mean() * periods_per_year
    downside_deviation = downside_ret.std() * np.sqrt(periods_per_year)
    return expected_return / downside_deviation

def calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss_price):
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

def calculate_drawdown(equity_curve):
    """
    equity_curve: pandas Series of portfolio value over time
    """
    hwm = equity_curve.cummax()
    drawdowns = (equity_curve - hwm) / hwm
    max_drawdown = drawdowns.min()
    return drawdowns, max_drawdown

def backtest_strategy(signals_df, initial_cash=100000, risk_per_trade=0.01):
    # Initialize portfolio
    cash = initial_cash
    portfolio_value = initial_cash
    positions = {}  # key: symbol, value: position details
    portfolio_history = []
    orders = []
    trades = []
    dates = sorted(signals_df['Date'].unique())
    for current_date in dates:
        daily_signals = signals_df[signals_df['Date'] == current_date]
        # First, update existing positions
        for symbol in list(positions.keys()):
            position = positions[symbol]
            position_df = daily_signals[daily_signals['Symbol'] == symbol]
            if not position_df.empty:
                close_price = position_df.iloc[0]['Close']
                # Check for stop loss or take profit
                if position['position'] == 'long':
                    if close_price <= position['stop_loss']:
                        # Stop loss hit, close position
                        cash += position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Sell', 'Shares': position['shares'], 'Reason': 'Stop Loss'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (close_price - trade['Entry Price']) * position['shares']
                        trade['Return'] = (close_price - trade['Entry Price']) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = position_df.iloc[0].to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                        del positions[symbol]
                    elif close_price >= position['take_profit']:
                        # Take profit hit, close position
                        cash += position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Sell', 'Shares': position['shares'], 'Reason': 'Take Profit'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (close_price - trade['Entry Price']) * position['shares']
                        trade['Return'] = (close_price - trade['Entry Price']) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = position_df.iloc[0].to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                        del positions[symbol]
                elif position['position'] == 'short':
                    if close_price >= position['stop_loss']:
                        # Stop loss hit, close position
                        cash -= position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Buy to Cover', 'Shares': position['shares'], 'Reason': 'Stop Loss'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (trade['Entry Price'] - close_price) * position['shares']
                        trade['Return'] = (trade['Entry Price'] - close_price) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = position_df.iloc[0].to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                        del positions[symbol]
                    elif close_price <= position['take_profit']:
                        # Take profit hit, close position
                        cash -= position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Buy to Cover', 'Shares': position['shares'], 'Reason': 'Take Profit'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (trade['Entry Price'] - close_price) * position['shares']
                        trade['Return'] = (trade['Entry Price'] - close_price) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = position_df.iloc[0].to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                        del positions[symbol]
            else:
                # Update position value
                # Assuming we have data for the symbol
                position_data = signals_df[(signals_df['Symbol'] == symbol) & (signals_df['Date'] == current_date)]
                if not position_data.empty:
                    close_price = position_data.iloc[0]['Close']
                    position['current_price'] = close_price
                    positions[symbol] = position
        # Now process new signals
        for idx, row in daily_signals.iterrows():
            symbol = row['Symbol']
            signal = row['Predicted_Label']
            if symbol in positions:
                # Already have a position
                position = positions[symbol]
                if (position['position'] == 'long' and signal == -1) or (position['position'] == 'short' and signal == 1):
                    # Close existing position due to opposite signal
                    close_price = row['Close']
                    if position['position'] == 'long':
                        cash += position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Sell', 'Shares': position['shares'], 'Reason': 'Opposite Signal'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (close_price - trade['Entry Price']) * position['shares']
                        trade['Return'] = (close_price - trade['Entry Price']) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = row.to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                    elif position['position'] == 'short':
                        cash -= position['shares'] * close_price
                        orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Buy to Cover', 'Shares': position['shares'], 'Reason': 'Opposite Signal'})
                        # Complete the trade
                        trade = position['trade']
                        trade['Exit Date'] = current_date
                        trade['Exit Price'] = close_price
                        trade['Profit/Loss'] = (trade['Entry Price'] - close_price) * position['shares']
                        trade['Return'] = (trade['Entry Price'] - close_price) / trade['Entry Price']
                        trade['Duration'] = (trade['Exit Date'] - trade['Entry Date']).days
                        trade['Outcome'] = 'Loss' if trade['Profit/Loss'] < 0 else 'Win'
                        # Record features at entry
                        trade_features = row.to_dict()
                        trade.update(trade_features)
                        trades.append(trade)
                    del positions[symbol]
                    # Proceed to open new position below if applicable
                else:
                    continue  # Ignore signal if same as existing position
            # Open new positions
            if signal == 1 and symbol not in positions:
                # Long entry
                entry_price = row['Entry_Price']
                stop_loss = row['Stop_Loss']
                take_profit = row['Take_Profit']
                shares = calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss)
                if shares == 0:
                    continue
                total_cost = shares * entry_price
                if cash >= total_cost:
                    cash -= total_cost
                    trade = {
                        'Symbol': symbol,
                        'Entry Date': current_date,
                        'Entry Price': entry_price,
                        'Position': 'long',
                        'Shares': shares,
                        'Predicted_Label': signal,
                        # Record features at entry
                        **row.to_dict()
                    }
                    positions[symbol] = {'position': 'long', 'shares': shares, 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'current_price': entry_price, 'trade': trade}
                    orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Buy', 'Shares': shares, 'Reason': 'New Long Position'})
            elif signal == -1 and symbol not in positions:
                # Short entry
                entry_price = row['Entry_Price']
                stop_loss = row['Stop_Loss']
                take_profit = row['Take_Profit']
                shares = calculate_position_size(portfolio_value, risk_per_trade, entry_price, stop_loss)
                if shares == 0:
                    continue
                total_credit = shares * entry_price
                cash += total_credit  # Short selling, receive cash
                trade = {
                    'Symbol': symbol,
                    'Entry Date': current_date,
                    'Entry Price': entry_price,
                    'Position': 'short',
                    'Shares': shares,
                    'Predicted_Label': signal,
                    # Record features at entry
                    **row.to_dict()
                }
                positions[symbol] = {'position': 'short', 'shares': shares, 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit, 'current_price': entry_price, 'trade': trade}
                orders.append({'Date': current_date, 'Symbol': symbol, 'Action': 'Sell Short', 'Shares': shares, 'Reason': 'New Short Position'})
        # Update portfolio value
        holdings_value = 0
        for symbol, position in positions.items():
            holdings_value += position['shares'] * position['current_price']
        portfolio_value = cash + holdings_value
        portfolio_history.append({'Date': current_date, 'Portfolio Value': portfolio_value})
    # Create DataFrame for portfolio history
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    portfolio_df.set_index('Date', inplace=True)
    # Calculate daily returns
    portfolio_df['Returns'] = portfolio_df['Portfolio Value'].pct_change()
    # Calculate drawdowns
    drawdowns, max_drawdown = calculate_drawdown(portfolio_df['Portfolio Value'])
    portfolio_df['Drawdown'] = drawdowns
    # Calculate performance metrics
    total_return = portfolio_df['Portfolio Value'][-1] / initial_cash - 1
    annualized_return = annualize_rets(portfolio_df['Returns'].dropna())
    annualized_vol = annualize_vol(portfolio_df['Returns'].dropna())
    sharpe_ratio = sharpe_ratio_func(portfolio_df['Returns'].dropna())
    sortino_ratio = sortino_ratio_func(portfolio_df['Returns'].dropna())
    # Print performance metrics
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Annualized Return: {annualized_return*100:.2f}%")
    print(f"Annualized Volatility: {annualized_vol*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    # Create DataFrame for trades
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
        trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
        # Calculate additional metrics
        total_trades = len(trades_df)
        num_winning_trades = len(trades_df[trades_df['Profit/Loss'] > 0])
        num_losing_trades = len(trades_df[trades_df['Profit/Loss'] <= 0])
        win_rate = num_winning_trades / total_trades if total_trades > 0 else np.nan
        average_win = trades_df[trades_df['Profit/Loss'] > 0]['Profit/Loss'].mean()
        average_loss = trades_df[trades_df['Profit/Loss'] <= 0]['Profit/Loss'].mean()
        average_win_pct = trades_df[trades_df['Profit/Loss'] > 0]['Return'].mean() * 100
        average_loss_pct = trades_df[trades_df['Profit/Loss'] <= 0]['Return'].mean() * 100
        average_duration = trades_df['Duration'].mean()
        trades_per_day = total_trades / ((portfolio_df.index.max() - portfolio_df.index.min()).days + 1)
        best_trade = trades_df.loc[trades_df['Profit/Loss'].idxmax()]
        worst_trade = trades_df.loc[trades_df['Profit/Loss'].idxmin()]
        grouped_by_symbol = trades_df.groupby('Symbol')['Profit/Loss'].sum()
        most_profitable_stock = grouped_by_symbol.idxmax()
        most_loss_stock = grouped_by_symbol.idxmin()
        # Print additional metrics
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {num_winning_trades}")
        print(f"Losing Trades: {num_losing_trades}")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Average Win: ${average_win:.2f}")
        print(f"Average Loss: ${average_loss:.2f}")
        print(f"Average Winning Trade Return: {average_win_pct:.2f}%")
        print(f"Average Losing Trade Return: {average_loss_pct:.2f}%")
        print(f"Average Trade Duration: {average_duration:.2f} days")
        print(f"Trades per Day: {trades_per_day:.2f}")
        print(f"Best Trade Profit: ${best_trade['Profit/Loss']:.2f} ({best_trade['Symbol']})")
        print(f"Worst Trade Loss: ${worst_trade['Profit/Loss']:.2f} ({worst_trade['Symbol']})")
        print(f"Most Profitable Stock: {most_profitable_stock} with Profit ${grouped_by_symbol[most_profitable_stock]:.2f}")
        print(f"Stock with Most Loss: {most_loss_stock} with Loss ${grouped_by_symbol[most_loss_stock]:.2f}")
    else:
        print("No trades were made during the backtest.")
    # Plot equity curve and drawdowns
    plt.figure(figsize=(14,7))
    plt.subplot(2,1,1)
    plt.plot(portfolio_df['Portfolio Value'])
    plt.title('Equity Curve')
    plt.subplot(2,1,2)
    plt.plot(portfolio_df['Drawdown'])
    plt.title('Drawdowns')
    plt.tight_layout()
    plt.show()
    # Save orders to CSV
    orders_df = pd.DataFrame(orders)
    orders_df.to_csv(DATA_DIR / 'backtest_orders.csv', index=False)
    trades_df.to_csv(DATA_DIR / 'backtest_trades.csv', index=False)
    print(f"Backtest completed. Final Portfolio Value: ${portfolio_value:.2f}")
    print(f"Orders saved to backtest_orders.csv")
    print(f"Trades saved to backtest_trades.csv")
    return portfolio_df, orders_df, trades_df

if __name__ == "__main__":
    # Load signals
    if os.path.exists(DATA_DIR / 'buy_signals.csv') and os.path.exists(DATA_DIR / 'sell_signals.csv'):
        buy_signals = pd.read_csv(DATA_DIR / 'buy_signals.csv')
        sell_signals = pd.read_csv(DATA_DIR / 'sell_signals.csv')
        signals_df = pd.concat([buy_signals, sell_signals], ignore_index=True)
        # Combine signals and sort by date
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        signals_df.sort_values(by='Date', inplace=True)
        # Run backtest
        portfolio_df, orders_df, trades_df = backtest_strategy(signals_df)
    else:
        print("Buy and sell signals files not found.")