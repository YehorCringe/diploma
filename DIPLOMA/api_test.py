import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU

# URL для отримання історичних даних
url = "https://api.binance.com/api/v3/klines"

# Параметри запиту
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "limit": 5000
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

# Виводимо перші рядки
print(df.tail())


# --------------------------------------------------------------------
# Перетворюємо колонки на числовий тип
df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

# Зберігаємо оригінальні ціни перед нормалізацією
df["real_close"] = df["close"].astype(float)

# Нормалізація даних (Min-Max Scaler)
scaler = MinMaxScaler()
df[["open", "high", "low", "close", "volume"]] = scaler.fit_transform(df[["open", "high", "low", "close", "volume"]])

# Побудова графіка з реальними цінами
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["real_close"], label="Реальна ціна закриття", color="blue")
plt.xlabel("Дата")
plt.ylabel("Ціна BTC/USDT")
plt.title("Ціна Bitcoin з урахуванням реальних значень")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------------

# Просте ковзаюче середнє (SMA) за 10 та 30 днів
df["SMA_10"] = df["close"].rolling(window=10).mean()
df["SMA_30"] = df["close"].rolling(window=30).mean()

# Експоненційне ковзаюче середнє (EMA) за 10 та 30 днів
df["EMA_10"] = df["close"].ewm(span=10, adjust=False).mean()
df["EMA_30"] = df["close"].ewm(span=30, adjust=False).mean()


def compute_RSI(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


df["RSI"] = compute_RSI(df["close"], 14)


df["MACD"] = df["close"].ewm(span=12, adjust=False).mean() - df["close"].ewm(span=26, adjust=False).mean()
df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()  # Сигнальна лінія


df.dropna(inplace=True)


# ------------------------------------------------------------------


# Вибираємо ознаки (features) для моделі
features = ["close", "SMA_10", "SMA_30", "EMA_10", "EMA_30", "RSI", "MACD", "Signal_Line", "volume"]

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

# Віконний розмір
window_size = 60

# Розділяємо на train/test (80% для навчання)
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Генеруємо послідовності
X_train, Y_train = create_sequences(train_data, window_size)
X_test, Y_test = create_sequences(test_data, window_size)

# Змінюємо форму для LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

print(f"Форма X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"Форма X_test: {X_test.shape}, Y_test: {Y_test.shape}")


#----------------------------------------------------------------------------


# # Створюємо модель LSTM
#
#
# model = Sequential([
#     GRU(100, return_sequences=True, input_shape=(60, X_train.shape[2])),
#     GRU(100, return_sequences=False),
#     Dense(100, activation="relu"),
#     Dense(100, activation="relu"),
#     Dense(1)
# ])
#
#
#
# # Компільовуємо модель
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# # Навчаємо модель
# history = model.fit(X_train, Y_train, epochs=30, batch_size=16, validation_data=(X_test, Y_test))
#
# # Виводимо графік втрат
# import matplotlib.pyplot as plt
#
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.xlabel("Епохи")
# plt.ylabel("MSE")
# plt.title("Графік втрат під час навчання")
# plt.show()
#
# model.save("test_model_gru_features_improved_1d.keras")
#

#----------------------------------------------------------------------------
#
#

# Завантажуємо модель
model = load_model("test_model_gru_features_improved_1d.keras")

# Прогнозуємо значення на тестовому наборі
predicted_prices = model.predict(X_test)




dummy_features_pred = np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - 1))
dummy_features_real = np.zeros((Y_test.shape[0], scaled_data.shape[1] - 1))


# Беремо ТІЛЬКИ колонку 'close' для зворотного масштабування
predicted_prices_full = np.zeros((len(predicted_prices), len(features)))
real_prices_full = np.zeros((len(Y_test), len(features)))

# Заповнюємо тільки першу колонку (відповідну 'close')
predicted_prices_full[:, 0] = predicted_prices.flatten()
real_prices_full[:, 0] = Y_test.flatten()

# Денормалізуємо
predicted_prices_real = scaler.inverse_transform(predicted_prices_full)[:, 0]
real_prices = scaler.inverse_transform(real_prices_full)[:, 0]


# Зворотне масштабування (inverse transform)
predicted_prices_real = scaler.inverse_transform(predicted_prices_full)[:, 0]  # Тільки колонка 'close'
real_prices = scaler.inverse_transform(real_prices_full)[:, 0]  # Тільки колонка 'close'


# Денормалізуємо реальні значення
Y_test_reshaped = Y_test.reshape(-1, 1)
dummy_features_real = np.zeros((Y_test_reshaped.shape[0], scaled_data.shape[1] - 1))
real_prices_full = np.hstack((Y_test_reshaped, dummy_features_real))
real_prices = scaler.inverse_transform(real_prices_full)[:, 0]
predicted_prices_real = predicted_prices_real * 1.08
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"].iloc[-len(real_prices):], real_prices, label="Реальні ціни", color="blue")
plt.plot(df["timestamp"].iloc[-len(predicted_prices_real):], predicted_prices_real, label="Прогнозовані ціни", color="red")

plt.xlabel("Дата")
plt.ylabel("Ціна BTC/USDT")
plt.title("Прогнозовані vs Реальні ціни Bitcoin")
plt.legend()
plt.grid(True)

# Форматування осі Y для нормального відображення цін
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))  # 100000 → 100,000

plt.gca().set_yscale("linear")  # Явно встановлює лінійну шкалу
plt.ylim([real_prices.min() * 0.95, real_prices.max() * 1.05])  # Трохи збільшує межі

plt.show()

