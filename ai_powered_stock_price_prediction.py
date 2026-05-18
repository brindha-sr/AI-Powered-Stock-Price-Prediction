import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Download 3 years of Apple stock data
STOCK = 'AAPL'
df = yf.download(STOCK, start='2021-01-01', end='2024-01-01', progress=False)
df = df[['Open','High','Low','Close','Volume']]
df.dropna(inplace=True)

print(f'✅ Downloaded {len(df)} trading days of {STOCK} data')
print(f'   Date range : {df.index[0].date()} → {df.index[-1].date()}')
print(f'   Price range: ${df["Close"].min().item():.2f} → ${df["Close"].max().item():.2f}')
df.tail(5)

close = df['Close'].squeeze()

df['MA_20']  = close.rolling(window=20).mean()   # 20-day average
df['MA_50']  = close.rolling(window=50).mean()   # 50-day average

delta   = close.diff()
gain    = delta.clip(lower=0)
loss    = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# 3. Bollinger Bands — upper/lower price boundaries
df['BB_mid']   = close.rolling(20).mean()
df['BB_upper'] = df['BB_mid'] + 2 * close.rolling(20).std()
df['BB_lower'] = df['BB_mid'] - 2 * close.rolling(20).std()

# 4. Price change % from previous day
df['Price_Change'] = close.pct_change() * 100

# Drop rows where indicators couldn't be calculated (first 50 days)
df.dropna(inplace=True)

print('✅ Technical indicators added!')
print(f'   Features created: MA_20, MA_50, RSI, Bollinger Bands, Price_Change')
print(f'   Rows remaining  : {len(df)}')
df[['Close','MA_20','MA_50','RSI','BB_upper','BB_lower']].tail(5)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Prepare data ---
# We use Close price + technical indicators as input features
FEATURES = ['Close', 'MA_20', 'MA_50', 'RSI', 'BB_upper', 'BB_lower', 'Price_Change']
data = df[FEATURES].values

# Scale all values between 0 and 1 (neural networks learn better this way)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# We only scale the Close column separately for inverse transform later
close_scaler = MinMaxScaler()
close_scaler.fit_transform(df[['Close']].values)

# --- Create sequences ---
# The model looks at the last WINDOW days to predict the next day
WINDOW = 30  # look back 30 trading days

X, y = [], []
for i in range(WINDOW, len(data_scaled)):
    X.append(data_scaled[i-WINDOW:i])   # last 30 days of all features
    y.append(data_scaled[i, 0])          # next day's Close price (index 0)

X, y = np.array(X), np.array(y)

# --- Train / Test split (80% train, 20% test) ---
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'✅ Data prepared')
print(f'   Training samples : {len(X_train)}')
print(f'   Testing  samples : {len(X_test)}')
print(f'   Input shape      : {X_train.shape}  (samples, days, features)')

# --- Build the LSTM model ---
model = Sequential([
    Input(shape=(WINDOW, len(FEATURES))),
    LSTM(64, return_sequences=True),   # first LSTM layer
    Dropout(0.2),                       # dropout prevents overfitting
    LSTM(32, return_sequences=False),  # second LSTM layer
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)                            # output: next day's price
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Early stopping — stops training if model stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print('\n🚀 Training started...')
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
print('\n✅ Training complete!')
# --- Training loss chart ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history['loss'],     label='Training Loss',   color='steelblue', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', color='coral',     linewidth=2)
ax.set_title('Model Training — Loss over Epochs', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
print('Lower loss = better model. Both lines going down = model is learning!')

# --- Make predictions on test set ---
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse scale: convert numbers back to actual dollar prices
def inverse_close(scaled_values):
    dummy = np.zeros((len(scaled_values), len(FEATURES)))
    dummy[:, 0] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_pred_actual = inverse_close(y_pred_scaled)
y_test_actual = inverse_close(y_test.reshape(-1, 1))

# --- Evaluation metrics ---
mae  = np.mean(np.abs(y_test_actual - y_pred_actual))
rmse = np.sqrt(np.mean((y_test_actual - y_pred_actual)**2))
mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

actual_dir   = np.diff(y_test_actual) > 0
predicted_dir = np.diff(y_pred_actual) > 0
dir_accuracy  = np.mean(actual_dir == predicted_dir) * 100

print('=' * 45)
print('      MODEL PERFORMANCE (Test Set)')
print('=' * 45)
print(f'  MAE  — avg error in dollars : ${mae:.2f}')
print(f'  RMSE — stricter error        : ${rmse:.2f}')
print(f'  MAPE — error as percentage   : {mape:.1f}%')
print(f'  Direction Accuracy           : {dir_accuracy:.1f}%')
print('=' * 45)
print(f'\n  Direction accuracy above 55% is considered good for stocks.')
print(f'  Random guessing would give only 50%.')
# Get the test period dates and actual data
test_dates  = df.index[split + WINDOW:]
test_close  = df['Close'].values[split + WINDOW:]
test_rsi    = df['RSI'].values[split + WINDOW:]

# --- Signal Logic ---
# We combine 2 things:
#   1. LSTM prediction: is the predicted price HIGHER than today? (model says price will rise)
#   2. RSI: is the stock oversold (RSI < 35)? or overbought (RSI > 65)?

signals = []
for i in range(len(y_pred_actual)):
    predicted_price = y_pred_actual[i]
    current_price   = test_close[i]
    rsi             = test_rsi[i]

    lstm_says_rise = predicted_price > current_price
    lstm_says_fall = predicted_price < current_price

    # BUY: model predicts rise AND RSI shows oversold
    if lstm_says_rise and rsi < 45:
        signals.append('BUY')
    # SELL: model predicts fall AND RSI shows overbought
    elif lstm_says_fall and rsi > 55:
        signals.append('SELL')
    else:
        signals.append('HOLD')

signals = np.array(signals)

buy_count  = np.sum(signals == 'BUY')
sell_count = np.sum(signals == 'SELL')
hold_count = np.sum(signals == 'HOLD')

print('✅ Buy/Sell Signals generated!')
print(f'   BUY  signals: {buy_count}')
print(f'   SELL signals: {sell_count}')
print(f'   HOLD signals: {hold_count}')

# --- Plot: Actual vs Predicted + Buy/Sell signals ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [3, 1]})

# Top chart: prices + signals
ax1.plot(test_dates, test_close,     label='Actual Price',     color='#2c3e50', linewidth=2)
ax1.plot(test_dates, y_pred_actual,  label='Predicted Price',  color='steelblue', linewidth=1.5, linestyle='--', alpha=0.9)

# Plot BUY signals as green triangles pointing up
buy_idx  = np.where(signals == 'BUY')[0]
sell_idx = np.where(signals == 'SELL')[0]

ax1.scatter(test_dates[buy_idx],  test_close[buy_idx],
            color='#27ae60', marker='^', s=100, zorder=5, label='BUY Signal')
ax1.scatter(test_dates[sell_idx], test_close[sell_idx],
            color='#e74c3c', marker='v', s=100, zorder=5, label='SELL Signal')

ax1.set_title(f'{STOCK} — LSTM Predictions + AI Buy/Sell Signals', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylabel('Price (USD)')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Bottom chart: RSI
ax2.plot(test_dates, test_rsi, color='purple', linewidth=1.5, label='RSI')
ax2.axhline(y=70, color='red',   linestyle='--', alpha=0.6, linewidth=1, label='Overbought (70)')
ax2.axhline(y=30, color='green', linestyle='--', alpha=0.6, linewidth=1, label='Oversold (30)')
ax2.fill_between(test_dates, test_rsi, 70, where=(test_rsi > 70), alpha=0.15, color='red')
ax2.fill_between(test_dates, test_rsi, 30, where=(test_rsi < 30), alpha=0.15, color='green')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_signals.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved as predictions_signals.png')
# Take the last WINDOW days of data as input
last_window = data_scaled[-WINDOW:].copy()  # shape: (30, 7)

future_predictions = []
current_window     = last_window.copy()

for day in range(7):
    # Reshape to (1, WINDOW, features)
    input_seq = current_window.reshape(1, WINDOW, len(FEATURES))

    # Predict next day (scaled value)
    next_day_scaled = model.predict(input_seq, verbose=0)[0, 0]
    future_predictions.append(next_day_scaled)

    # Slide the window: drop day 0, add the new prediction at the end
    new_row         = current_window[-1].copy()  # copy last row
    new_row[0]      = next_day_scaled             # update Close with prediction
    current_window  = np.vstack([current_window[1:], new_row])

# Convert scaled predictions back to actual dollar prices
future_prices = inverse_close(np.array(future_predictions).reshape(-1, 1))

# Create future dates (skip weekends)
from pandas.tseries.offsets import BDay
last_date    = df.index[-1]
future_dates = pd.date_range(start=last_date + BDay(1), periods=7, freq='B')

print('✅ 7-Day Future Forecast Generated!')
print(f'\n   Last known price : ${df["Close"].iloc[-1].item():.2f}  ({last_date.date()})')
print()
print('   Day | Date          | Predicted Price | Change')
print('   ' + '-'*50)
prev = df['Close'].iloc[-1].item()
for i, (date, price) in enumerate(zip(future_dates, future_prices)):
    change = price - prev
    arrow  = '▲' if change >= 0 else '▼'
    print(f'   {i+1:>2}  | {str(date.date()):13} | ${price:>10.2f}      | {arrow} ${abs(change):.2f}')
    prev = price

# --- Plot the 7-day forecast ---
# Show last 60 days of actual data + 7 days forecast
recent_dates  = df.index[-60:]
recent_prices = df['Close'].values[-60:]

fig, ax = plt.subplots(figsize=(13, 5))

# Historical prices
ax.plot(recent_dates, recent_prices,
        color='#2c3e50', linewidth=2, label='Historical Price')

# Connecting line from last actual to first forecast
ax.plot([recent_dates[-1], future_dates[0]],
        [recent_prices[-1][0], future_prices[0]],
        color='steelblue', linewidth=1.5, linestyle='--')

# Forecast line
ax.plot(future_dates, future_prices,
        color='steelblue', linewidth=2.5, linestyle='--', marker='o',
        markersize=7, markerfacecolor='white', markeredgewidth=2,
        label='7-Day Forecast')

# Uncertainty shading (±2% band around forecast)
upper = future_prices * 1.02
lower = future_prices * 0.98
ax.fill_between(future_dates, lower, upper, alpha=0.12, color='steelblue', label='±2% Uncertainty')

# Label each forecast point
for i, (date, price) in enumerate(zip(future_dates, future_prices)):
    ax.annotate(f'${price:.1f}',
                xy=(date, price),
                xytext=(0, 14), textcoords='offset points',
                ha='center', fontsize=8.5, color='steelblue', fontweight='bold')

# Vertical divider between actual and forecast
ax.axvline(x=recent_dates[-1], color='gray', linestyle=':', alpha=0.6)
ax.text(recent_dates[-1], ax.get_ylim()[0],
        '  Today', fontsize=9, color='gray', va='bottom')

ax.set_title(f'{STOCK} — 7-Day Future Price Forecast (LSTM)', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('forecast_7days.png', dpi=150, bbox_inches='tight')
plt.show()
print('Chart saved as forecast_7days.png')
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f'{STOCK} Stock Price Prediction — ML Dashboard', fontsize=16, fontweight='bold', y=0.98)

# ── Chart 1: Full actual vs predicted ───────────────
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(test_dates, test_close,    color='#2c3e50',   linewidth=1.5, label='Actual')
ax1.plot(test_dates, y_pred_actual, color='steelblue', linewidth=1.2, linestyle='--', label='Predicted', alpha=0.8)
ax1.scatter(test_dates[buy_idx],  test_close[buy_idx],  color='#27ae60', marker='^', s=40, zorder=4, label='BUY')
ax1.scatter(test_dates[sell_idx], test_close[sell_idx], color='#e74c3c', marker='v', s=40, zorder=4, label='SELL')
ax1.set_title('Predictions + Signals', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.2)

# ── Chart 2: 7-day forecast ─────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df.index[-30:], df['Close'].values[-30:], color='#2c3e50', linewidth=2, label='Recent Price')
ax2.plot(future_dates,   future_prices, color='steelblue', linewidth=2, linestyle='--',
         marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2, label='7-Day Forecast')
ax2.fill_between(future_dates, future_prices*0.98, future_prices*1.02, alpha=0.15, color='steelblue')
ax2.set_title('7-Day Future Forecast', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

# ── Chart 3: Metrics bar chart ──────────────────────
ax3 = fig.add_subplot(2, 2, 3)
metric_names  = ['MAE ($)', 'RMSE ($)', 'MAPE (%)', 'Dir. Acc (%)']
metric_values = [mae, rmse, mape, dir_accuracy]
colors        = ['#3498db','#9b59b6','#e67e22','#27ae60']
bars = ax3.bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, metric_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_title('Model Evaluation Metrics', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(metric_values) * 1.2)

# ── Chart 4: Signal distribution pie ──────────────────────
ax4 = fig.add_subplot(2, 2, 4)
sizes  = [buy_count, sell_count, hold_count]
labels = [f'BUY\n{buy_count}', f'SELL\n{sell_count}', f'HOLD\n{hold_count}']
pie_colors = ['#27ae60', '#e74c3c', '#95a5a6']
wedges, texts, autotexts = ax4.pie(
    sizes, labels=labels, colors=pie_colors,
    autopct='%1.0f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
for t in autotexts: t.set_fontsize(10); t.set_fontweight('bold')
ax4.set_title('Signal Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Dashboard saved as dashboard.png')

# ── Final Summary ──────────────────────────────────────────
print('='*55)
print(' SUMMARY')
print('='*55)
print(f'  Stock          : {STOCK} (Apple Inc.)')
print(f'  Data           : {len(df)} trading days (2021–2024)')
print(f'  Model          : LSTM Neural Network')
print(f'  Features used  : Close, MA20, MA50, RSI, Bollinger Bands')
print(f'  Window size    : {WINDOW} days')
print()
print('  MODEL PERFORMANCE')
print(f'  MAE              : ${mae:.2f}')
print(f'  RMSE             : ${rmse:.2f}')
print(f'  MAPE             : {mape:.1f}%')
print(f'  Direction Acc.   : {dir_accuracy:.1f}%')
print()
print('  AI FEATURES')
print(f'  Buy/Sell Signals : {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD')
print(f'  7-Day Forecast   : ${future_prices[0]:.2f} → ${future_prices[-1]:.2f}')
print('='*55)
