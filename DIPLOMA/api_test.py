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
df[["open", "high", "low", "close", "volume"]] = scaler.fit_transform(df[["open", "high", "low", "close", "volume"]])


# ------------------------------------------------------------------

def compute_RSI(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))
    return RSI


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


# Створюємо модель LSTM

# model = Sequential([
#     Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, len(features))),
#     Dropout(0.2),
#     GRU(64, return_sequences=True),
#     Dropout(0.2),
#     GRU(64),
#
#     Dense(forecast_horizon, activation="linear"),
# ])
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
model = load_model("GRUCNNrecursive.keras")


predicted_scaled = model.predict(X_test)  # shape: (N, 7)

# Тільки першу ознаку (наприклад, close) інвертуємо
forecast_padded = np.zeros((len(predicted_scaled.flatten()), len(features)))
forecast_padded[:, 0] = predicted_scaled.flatten()

forecast_actual = scaler.inverse_transform(forecast_padded)[:, 0]

# plt.plot(range(len(prices[-60:])), prices[-60:], label="Реальні ціни")
plt.plot(range(len(prices[-60:]), len(prices[-60:]) + 7), forecast_actual[-7:], label="Прогноз на 7 днів")
plt.legend()
plt.title("Прямий прогноз на тиждень")
plt.show()

#----------------------------------------------------------------------------
#Прогнозуємо значення на тестовому наборі
# predicted_prices = model.predict(X_test)


# Рекурсивний прогноз
# def recursive_forecast(model, last_sequence, n_days):
#     """
#     Рекурсивне прогнозування на n днів уперед на основі останньої послідовності.
#
#     :param model: Навчена LSTM модель
#     :param last_sequence: остання послідовність (наприклад, shape = (50, 1))
#     :param n_days: кількість днів для прогнозу
#     :return: масив з n_days передбаченнями
#     """
#     forecast = []
#     current_input = last_sequence.copy()  # Наприклад, shape = (50, 1)
#
#     for _ in range(n_days):
#         # Додаємо осі batch і time для подачі в модель: (1, 50, 1)
#         input_reshaped = current_input.reshape(1, current_input.shape[0], current_input.shape[1])
#
#         # Прогнозуємо наступне значення
#         next_pred = model.predict(input_reshaped, verbose=0)
#
#         # Зберігаємо прогноз
#         forecast.append(next_pred[0][0])  # витягуємо значення зі структури [[x]]
#
#         # Готуємо нову послідовність для наступного прогнозу
#         next_input = next_pred[0]  # форма: (1,)
#         current_input = np.concatenate((current_input[1:], [next_input]), axis=0)
#
#     return np.array(forecast)
#
#
# # Використання
# last_sequence = scaled_data[-window_size:]  # останнє вікно
# forecast_scaled = recursive_forecast(model, last_sequence, n_days=7)
# forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
#
# # Інверсія тільки для першої (close) ознаки
# min_val = scaler.data_min_[0]
# max_val = scaler.data_max_[0]
# # forecast_scaled має shape (7, 1)
# # Створюємо "пустий" масив зі значеннями 0, де лише перша колонка — ваш прогноз
# forecast_padded = np.zeros((len(forecast_scaled), scaler.scale_.shape[0]))
# forecast_padded[:, 0] = np.array(forecast_scaled).reshape(-1)  # або .squeeze()
#
# # Інверсія з урахуванням повної кількості фіч
# forecast_actual = scaler.inverse_transform(forecast_padded)[:, 0]  # Беремо лише ціну
#
#
# # Візуалізація
# plt.figure(figsize=(10,5))
# # plt.plot(prices[-60:], label="Реальні ціни")
# plt.plot(range(len(prices[-60:]), len(prices[-60:]) + 7), forecast_actual, label="Прогноз")
# plt.legend()
# plt.title("Рекурсивний прогноз на 7 днів")
# plt.show()
#
# print("Прогноз (масштабований):", forecast_scaled.flatten())
# print("Прогноз (інверсія):", forecast_actual)


# # ------------------30 days---------------------------
# predicted = model.predict(X_test)  # shape: (N, 30)
# predicted_real = []
# real_prices = []
#
# for i in range(len(predicted)):
#     pred_full = np.hstack((predicted[i].reshape(-1, 1), np.zeros((forecast_horizon, scaled_data.shape[1] - 1))))
#     real_full = np.hstack((Y_test[i].reshape(-1, 1), np.zeros((forecast_horizon, scaled_data.shape[1] - 1))))
#
#     predicted_real.append(scaler.inverse_transform(pred_full)[:, 0])
#     real_prices.append(scaler.inverse_transform(real_full)[:, 0])
#
#
#
# plt.figure(figsize=(10, 5))
# plt.plot(real_prices[-1], label="Реальні ціни")
# plt.plot(predicted_real[-1], label="Прогноз на 30 днів")
# plt.legend()
# plt.title("Прогноз на 30 днів вперед")
# plt.xlabel("Дні вперед")
# plt.ylabel("Ціна BTC")
# plt.grid(True)
# plt.show()

# ------------------30 days---------------------------
#
#
# dummy_features_pred = np.zeros((predicted_prices.shape[0], scaled_data.shape[1] - 1))
# dummy_features_real = np.zeros((Y_test.shape[0], scaled_data.shape[1] - 1))
#
#
# # Беремо ТІЛЬКИ колонку 'close' для зворотного масштабування
# predicted_prices_full = np.zeros((len(predicted_prices), len(features)))
# real_prices_full = np.zeros((len(Y_test), len(features)))
#
# # Заповнюємо тільки першу колонку (відповідну 'close')
# predicted_prices_full[:, 0] = predicted_prices.flatten()
# real_prices_full[:, 0] = Y_test.flatten()
#
# # Денормалізуємо
# predicted_prices_real = scaler.inverse_transform(predicted_prices_full)[:, 0]
# real_prices = scaler.inverse_transform(real_prices_full)[:, 0]
#
#
# # Зворотне масштабування (inverse transform)
# predicted_prices_real = scaler.inverse_transform(predicted_prices_full)[:, 0]  # Тільки колонка 'close'
# real_prices = scaler.inverse_transform(real_prices_full)[:, 0]  # Тільки колонка 'close'
#
#
# # Денормалізуємо реальні значення
# Y_test_reshaped = Y_test.reshape(-1, 1)
# dummy_features_real = np.zeros((Y_test_reshaped.shape[0], scaled_data.shape[1] - 1))
# real_prices_full = np.hstack((Y_test_reshaped, dummy_features_real))
# real_prices = scaler.inverse_transform(real_prices_full)[:, 0]
#
# predicted_prices_real = predicted_prices_real * 0.98
#
# plt.figure(figsize=(12, 6))
# plt.plot(df["timestamp"].iloc[-len(real_prices):], real_prices, label="Реальні ціни", color="blue")
# plt.plot(df["timestamp"].iloc[-len(predicted_prices_real):], predicted_prices_real, label="Прогнозовані ціни", color="red")
#
# plt.xlabel("Дата")
# plt.ylabel("Ціна BTC/USDT")
# plt.title("Прогнозовані vs Реальні ціни Bitcoin")
# plt.legend()
# plt.grid(True)
#
# # Форматування осі Y для нормального відображення цін
# plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))  # 100000 → 100,000
#
# plt.gca().set_yscale("linear")  # Явно встановлює лінійну шкалу
# plt.ylim([real_prices.min() * 0.95, real_prices.max() * 1.05])  # Трохи збільшує межі
#
# plt.show()

# ============== ПРОГНОЗ НА МАЙБУТНЄ ВІД СЬОГОДНІ ==============
#
# # Беремо останні 60 днів з нормалізованих даних
# last_sequence = scaled_data[-window_size:]
# last_sequence = last_sequence.reshape(1, window_size, len(features))  # (1, 60, 9)
#
# # Робимо прогноз на 30 днів
# future_prediction = model.predict(last_sequence)[0]  # shape: (30,)
#
# # Підготовка для зворотного масштабування (inverse transform)
# future_prediction_full = np.hstack((
#     future_prediction.reshape(-1, 1),
#     np.zeros((forecast_horizon, scaled_data.shape[1] - 1))
# ))
# future_prediction_real = scaler.inverse_transform(future_prediction_full)[:, 0]
#
# # Побудова дат для прогнозу
# last_date = df["timestamp"].iloc[-1]
# future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
#
# # Побудова графіка
# plt.figure(figsize=(12, 6))
# plt.plot(future_dates, future_prediction_real, label="Прогноз на 30 днів від сьогодні", color="orange")
# plt.xlabel("Дата")
# plt.ylabel("Ціна BTC/USDT")
# plt.title("Прогноз Bitcoin на 30 днів вперед")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()