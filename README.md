# Trading – Mean Reversion Strategy (Python)

This project implements a machine-learning based **mean-reversion trading strategy** using a full research-to-execution pipeline. It includes data collection (Yahoo Finance or Interactive Brokers), feature engineering, model training, signal generation, backtesting, and live trading signal export.

---

## 📁 Project Structure

mean-reversion-ml/

├─ data/                      # Price data, datasets, signals, backtests (auto-generated)

├─ models/                    # Trained model + encoders

├─ params/                    # Per-stock indicator settings and tuned hyperparameters

├─ plots/                     # Static and interactive chart outputs

├─ data_preparation.py        # All python scripts

├─ model_training.py

├─ signal_generation.py

├─ backtesting.py

├─ trade_analysis.py

├─ visualize_signals.py

└─ live_signal_generation.py

---

## 🚀 End-to-End Workflow

You can execute the full pipeline step-by-step.

---

### 1️⃣ Data Collection & Feature Engineering

Creates labeled dataset with technical indicators:

python src/data_preparation.py

Outputs:
- data/filtered_dataset.csv
- models/label_encoder.pkl
- Cache: data/stock_data.pkl or data/stock_data_ib.pkl

---

### 2️⃣ Model Training

Trains Random Forest with walk-forward validation and SMOTE:

python src/model_training.py

Outputs:
- models/best_model.pkl
- params/best_parameters.json

---

### 3️⃣ Trade Signal Generation

Applies trained model to infer long/short predictions:

python src/signal_generation.py

Outputs:
- data/buy_signals.csv
- data/sell_signals.csv

---

### 4️⃣ Backtesting

Evaluates P&L and risk statistics (Sharpe, Sortino, Drawdowns):

python src/backtesting.py

Outputs:
- data/backtest_orders.csv
- data/backtest_trades.csv
- Equity and drawdown plots

---

### 5️⃣ Trade Analysis

Win/loss feature attribution, symbol-based analytics:

python src/trade_analysis.py

---

### 6️⃣ Charting

Candlesticks + Bollinger Bands + RSI + Stoch with signals:

python src/visualize_signals.py --data data/filtered_dataset.csv

For interactive HTML charts:

python src/visualize_signals.py --interactive

Saves into:
- plots/ and plots/interactive/

---

### 7️⃣ Live Signal Monitoring (IBKR)

Generates fresh predictions from Interactive Brokers:

python src/live_signal_generation.py

Outputs:
- data/live_signals_all.csv
- data/live_signals_recent.csv

---

## ✅ Dependencies

pip install -r requirements.txt

Key packages:
- pandas, numpy, yfinance, ta
- scikit-learn, imbalanced-learn, joblib
- ib-insync (optional, for IBKR)
- matplotlib, seaborn, plotly

---

## ⚙️ Configuration

Interactive Brokers connection parameters can be adjusted in:

IB.connect('127.0.0.1', 7497, clientId=1)

To change historical IB data:

duration = '3 M'
bar_size = '1 hour'
what_to_show = 'TRADES'

Model tuning parameters:
- params/best_parameters.json

---

## 📊 Strategy Summary

- Style: Mean-reversion classification
- Indicators: RSI, Stoch %K/%D, MACD, Bollinger Bands, ATR, MA50/MA200, ROC, Z-Score, Williams %R, Volume Change
- Trading logic: ML predictions + ATR-based SL/TP
- Backtesting: Walk-forward evaluation
- Risk Management:
  - Position sizing from stop-loss distance
  - Max size cap per position

---

## 🔒 Disclaimer

This project is developed for research and educational purposes only.
Nothing here constitutes financial advice. Use responsibly.
