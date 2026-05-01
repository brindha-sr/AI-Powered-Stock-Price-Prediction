# AI-Powered Stock Price Prediction

An end-to-end deep learning project that uses an **LSTM (Long Short-Term Memory) neural network** to predict stock prices, generate AI-driven buy/sell signals, and produce a 7-day future price forecast — all with interactive visualisations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Model Architecture](#model-architecture)
- [Technical Indicators](#technical-indicators)
- [Performance Metrics](#performance-metrics)
- [Disclaimer](#disclaimer)

---

## Overview

This project downloads 3 years of historical stock data (default: **Apple Inc. / AAPL**), engineers several technical indicators, trains an LSTM model, and then:

1. Evaluates the model on a held-out test set.
2. Generates **BUY / SELL / HOLD** trading signals by combining LSTM predictions with RSI analysis.
3. Forecasts the next **7 trading days** of closing prices.
4. Displays a comprehensive **ML dashboard** with four summary charts.

---

## Features

| Capability | Details |
|---|---|
| Data download | Fetches OHLCV data via `yfinance` |
| Feature engineering | MA-20, MA-50, RSI-14, Bollinger Bands, daily % change |
| Deep learning model | 2-layer stacked LSTM with Dropout regularisation |
| Trading signals | LSTM + RSI hybrid logic (BUY / SELL / HOLD) |
| Future forecast | 7-day autoregressive rolling-window prediction |
| Visualisations | Training loss, predictions+signals, RSI, forecast, dashboard |

---

## Tech Stack

- **Python 3.8+**
- [yfinance](https://github.com/ranaroussi/yfinance) — market data download
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data manipulation
- [scikit-learn](https://scikit-learn.org/) — `MinMaxScaler` feature scaling
- [TensorFlow / Keras](https://www.tensorflow.org/) — LSTM model
- [Matplotlib](https://matplotlib.org/) — charting and dashboard

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/brindha-sr/AI-Powered-Stock-Price-Prediction.git
cd AI-Powered-Stock-Price-Prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow
```

---

## Usage

```bash
python ai_powered_stock_price_prediction.py
```

To change the stock ticker, edit the `STOCK` variable near the top of the script:

```python
STOCK = 'AAPL'   # e.g. 'MSFT', 'GOOGL', 'TSLA'
```

You can also adjust the look-back window:

```python
WINDOW = 30   # number of past trading days used per prediction
```

---

## Outputs

The script prints a live summary to the console and saves three image files:

| File | Description |
|---|---|
| `predictions_signals.png` | Actual vs predicted prices with BUY/SELL markers and RSI subplot |
| `forecast_7days.png` | 7-day future price forecast with ±2 % uncertainty band |
| `dashboard.png` | 2×2 ML dashboard (predictions, forecast, metrics bar chart, signal pie chart) |

---

## Model Architecture

```
Input  →  (WINDOW=30, features=7)
LSTM   →  64 units, return_sequences=True
Dropout → 0.2
LSTM   →  32 units, return_sequences=False
Dropout → 0.2
Dense  →  16 units, ReLU
Dense  →  1 unit  (predicted next-day close price)

Optimizer : Adam (lr=0.001)
Loss      : Mean Squared Error (MSE)
Callbacks : EarlyStopping (patience=5, restore_best_weights=True)
```

---

## Technical Indicators

| Indicator | Window | Purpose |
|---|---|---|
| MA-20 | 20 days | Short-term trend |
| MA-50 | 50 days | Medium-term trend |
| RSI-14 | 14 days | Overbought / oversold momentum |
| Bollinger Bands | 20 days ± 2σ | Volatility range |
| Price Change % | 1 day | Daily momentum |

---

## Performance Metrics

The model is evaluated on a 20 % held-out test set using:

- **MAE** — Mean Absolute Error (average dollar error)
- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAPE** — Mean Absolute Percentage Error
- **Direction Accuracy** — percentage of days where the predicted price movement direction (up/down) matches the actual direction. Values above **55 %** are considered meaningful for stock markets (random guessing = 50 %).

---

## Disclaimer

> This project is for **educational and research purposes only**. It is not financial advice. Stock markets are inherently unpredictable, and past model performance does not guarantee future results. Do not use these predictions to make real trading decisions.