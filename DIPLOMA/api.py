import requests
import pandas as pd
import numpy as np
from keras.src.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.losses import Huber
import requests

# URL для отримання історичних даних
url = "https://api.binance.com/api/v3/klines"

# Параметри запиту
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 10000
}

# Виконуємо запит
response = requests.get(url, params=params)
data = response.json()

# Конвертуємо в DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
])


# Перетворюємо timestamp у дату
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Залишаємо тільки важливі колонки
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# --------------------------------------------------------------------
# Перетворюємо колонки на числовий тип
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

# Зберігаємо оригінальні ціни перед нормалізацією
df["real_close"] = df["close"].astype(float)

# Нормалізація даних (Min-Max Scaler)
scaler = MinMaxScaler()
df[["open", "high", "low", "volume"]] = scaler.fit_transform(df[["open", "high", "low", "volume"]])


# ------------------------------------------------------------------

def compute_RSI(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


import requests

import requests

def get_btc_prices_from_exchanges():
    prices = {}

    # Binance
    try:
        binance_response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
        binance_data = binance_response.json()
        prices["Binance"] = round(float(binance_data["price"]), 2)
    except Exception as e:
        prices["Binance"] = f"Error: {e}"

    # Bybit (V5 API)
    try:
        bybit_response = requests.get("https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT")
        bybit_data = bybit_response.json()

        if isinstance(bybit_data, dict) and \
           "result" in bybit_data and \
           "list" in bybit_data["result"] and \
           len(bybit_data["result"]["list"]) > 0:

            ticker = bybit_data["result"]["list"][0]
            prices["Bybit"] = round(float(ticker["lastPrice"]), 2)
        else:
            prices["Bybit"] = "Unexpected format"
    except Exception as e:
        prices["Bybit"] = f"Error: {e}"

    return prices




# Просте ковзаюче середнє (SMA) за 10 та 30 днів
df["SMA_10"] = df["close"].rolling(window=10).mean()
df["SMA_30"] = df["close"].rolling(window=30).mean()

# Експоненційне ковзаюче середнє (EMA) за 10 та 30 днів
df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["EMA_30"] = df["close"].ewm(span=30, adjust=False).mean()

df["RSI"] = compute_RSI(df["close"], 14)

df["MACD"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()  # Сигнальна лінія

prices = df["real_close"].values

scaler_close = MinMaxScaler()
df["close_scaled"] = scaler_close.fit_transform(df[["close"]])


df.dropna(inplace=True)



# ------------------------------------------------------------------

# Вибираємо ознаки (features) для моделі
features = ["close_scaled", "SMA_10", "SMA_30", "EMA_10", "EMA_30", "RSI", "MACD", "Signal_Line", "volume"]

# Масштабуємо дані
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])

# Функція для створення послідовностей
def create_sequences(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :])
        Y.append(data[i + window_size, 0])
    return np.array(X), np.array(Y)


def create_multistep_sequences(data, window_size, forecast_horizon):
    X, Y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i + window_size, :])                    # Вхід
        Y.append(data[i + window_size:i + window_size + forecast_horizon, 0])  # Прогноз (тільки close)
    return np.array(X), np.array(Y)


# Віконний розмір
window_size = 120
forecast_horizon = 7

# Розділяємо на train/test (80% для навчання)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

X_train, Y_train = create_multistep_sequences(train_data, window_size, forecast_horizon)
X_test, Y_test = create_multistep_sequences(test_data, window_size, forecast_horizon)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))


print(f"Форма X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"Форма X_test: {X_test.shape}, Y_test: {Y_test.shape}")


#----------------------------------------------------------------------------


# Створюємо модель

model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, len(features))),
    Dropout(0.2),
    GRU(64, return_sequences=True),
    Dropout(0.2),
    GRU(64),

    Dense(forecast_horizon, activation="linear"),
])


#
#
#
#
# # Компільовуємо модель
# opt = SGD(learning_rate=0.1)
# model.compile(optimizer=opt, loss=Huber())
#
# # Навчаємо модель
# history = model.fit(X_train, Y_train, epochs=30, batch_size=16, validation_data=(X_test, Y_test))
#
# # Виводимо графік втрат
#
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.xlabel("Епохи")
# plt.ylabel("MSE")
# plt.title("Графік втрат під час навчання")
# plt.show()
#
# model.save("GRUCNNrecursive.keras")


#----------------------------------------------------------------------------


#Завантажуємо модель
# model = load_model("GRUCNNrecursive.keras")
#
#
# predicted_scaled = model.predict(X_test)  # shape: (N, 7)
#
# # Тільки першу ознаку (наприклад, close) інвертуємо
# forecast_padded = np.zeros((len(predicted_scaled.flatten()), len(features)))
# forecast_padded[:, 0] = predicted_scaled.flatten()
#
#
# last_prediction = predicted_scaled[-1]  # shape: (7,)
# forecast_actual = scaler_close.inverse_transform(last_prediction.reshape(-1, 1)).flatten()
#
# # Отримуємо останню відому реальну ціну
# last_real_price = prices[-1]
#
# # Перший прогнозований день
# first_forecast = forecast_actual[0]
#
# # Обчислюємо зсув
# delta = last_real_price - first_forecast
#
# # Коригуємо весь прогноз
# forecast_aligned = forecast_actual + delta
#
# plt.plot(range(len(prices[-60:])), prices[-60:], label="Реальні ціни")
# plt.plot(range(len(prices[-60:]), len(prices[-60:]) + 7), forecast_aligned, label="Прогноз на 7 днів")
# plt.legend()
# plt.title("Прямий прогноз на тиждень")
# plt.show()

#----------------------------------------------------------------------------
#Прогнозуємо значення на тестовому наборі
# predicted_prices = model.predict(X_test)

