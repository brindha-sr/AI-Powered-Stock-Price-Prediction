📈 Stock Price Prediction using LSTM 
🚀 Project Overview

This project builds a complete stock price prediction system using Long Short-Term Memory (LSTM) neural networks. It uses historical stock data along with technical indicators to:

Predict future stock prices
Generate BUY/SELL/HOLD signals
Forecast next 7 trading days
Visualize results with a dashboard

🧠 Key Features
📥 Data Acquisition
Downloads historical stock data using yfinance
Example: Apple stock (AAPL) from 2021–2024
⚙️ Feature Engineering

The model uses both price and technical indicators:

Moving Averages:
MA_20 (20-day)
MA_50 (50-day)
Relative Strength Index (RSI)
Bollinger Bands:
Upper Band
Lower Band
Price Change (%)

🧹 Data Preparation
Scales data using MinMaxScaler
Creates time-series sequences using a 30-day window
Splits dataset into:
80% Training
20% Testing

🤖 LSTM Model Architecture
Input Layer (30 days × 7 features)
↓
LSTM (64 units, return_sequences=True)
↓
Dropout (0.2)
↓
LSTM (32 units)
↓
Dropout (0.2)
↓
Dense (16, ReLU)
↓
Output Layer (1 neuron → Next day price)
Optimizer: Adam (learning rate = 0.001)
Loss: Mean Squared Error (MSE)
EarlyStopping used to prevent overfitting

📊 Model Evaluation

Metrics used:

MAE – Mean Absolute Error
RMSE – Root Mean Squared Error
MAPE – Mean Absolute Percentage Error
Directional Accuracy – Correct trend prediction
💡 Trading Signal Logic

Signals are generated using:

LSTM prediction (price up/down)
RSI indicator
Condition	Signal
Price ↑ & RSI < 45	BUY
Price ↓ & RSI > 55	SELL
Otherwise	HOLD

🔮 Future Forecasting
Predicts next 7 trading days
Uses recursive prediction (sliding window method)
Includes uncertainty band (±2%)

📉 Visualizations Generated
Training Loss Graph
Actual vs Predicted Prices
Buy/Sell Signal Chart
RSI Indicator Plot
7-Day Forecast Plot

Dashboard with:
Predictions
Forecast
Metrics
Signal distribution

🛠️ Technologies Used
Python
yfinance
pandas, numpy
matplotlib
scikit-learn
tensorflow (Keras)
