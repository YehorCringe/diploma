import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dropout, Dense
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import os
import joblib

def fetch_binance_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 5000
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def compute_indicators(df):
    df["SMA_10"] = df["close"].rolling(window=10).mean()
    df["SMA_30"] = df["close"].rolling(window=30).mean()
    df["EMA_10"] = df["close"].ewm(span=10).mean()
    df["EMA_30"] = df["close"].ewm(span=30).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()
    return df.dropna()

def create_multistep_sequences(data, window_size, forecast_horizon):
    X, Y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i + window_size])
        Y.append(data[i + window_size:i + window_size + forecast_horizon, 0])
    return np.array(X), np.array(Y)

def train_and_save_model():
    df = fetch_binance_data()
    df = compute_indicators(df)

    if len(df) < 200:
        print("❌ Недостатньо даних для тренування.")
        return

    scaler_close = MinMaxScaler()
    df["close_scaled"] = scaler_close.fit_transform(df[["close"]])
    joblib.dump(scaler_close, "models/scaler_close.pkl")

    features = ["close_scaled", "SMA_10", "SMA_30", "EMA_10", "EMA_30", "RSI", "MACD", "Signal_Line", "volume"]
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    joblib.dump(scaler, "models/feature_scaler.pkl")

    window_size = 120
    forecast_horizon = 7

    X, Y = create_multistep_sequences(scaled_data, window_size, forecast_horizon)
    Y_diff = Y - X[:, -1, 0:1]  # зміни ціни від останньої точки в X

    delta_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_scaled = delta_scaler.fit_transform(Y_diff.reshape(-1, 1)).reshape(Y_diff.shape)
    joblib.dump(delta_scaler, "models/delta_scaler.pkl")

    train_size = int(len(X) * 0.8)
    X_train, Y_train = X[:train_size], Y_scaled[:train_size]
    X_test, Y_test = X[train_size:], Y_scaled[train_size:]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(features))

    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, len(features))),
        Dropout(0.2),
        GRU(64, return_sequences=True, recurrent_dropout=0.2),
        Dropout(0.2),
        GRU(64, recurrent_dropout=0.2),
        Dense(forecast_horizon, activation="linear")
    ])

    model.compile(optimizer=SGD(learning_rate=0.015), loss=Huber())
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model_binance.keras")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History (Binance)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/loss_binance.png")
    print("✅ Модель збережено: models/model_binance.keras")

def run_forecast_pipeline(df, model, window_size=120, forecast_horizon=7):
    scaler_close = joblib.load("models/scaler_close.pkl")
    feature_scaler = joblib.load("models/feature_scaler.pkl")
    delta_scaler = joblib.load("models/delta_scaler.pkl")

    df = compute_indicators(df)
    df["close_scaled"] = scaler_close.transform(df[["close"]])

    features = ["close_scaled", "SMA_10", "SMA_30", "EMA_10", "EMA_30", "RSI", "MACD", "Signal_Line", "volume"]
    df.dropna(inplace=True)

    scaled_data = feature_scaler.transform(df[features])
    input_seq = scaled_data[-window_size:]
    input_seq = input_seq.reshape(1, window_size, len(features))

    forecast_scaled = model.predict(input_seq)[0]
    forecast_deltas = delta_scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

    last_scaled_close = scaled_data[-1, 0]
    forecast_scaled_close = last_scaled_close + forecast_deltas

    forecast_prices = scaler_close.inverse_transform(forecast_scaled_close.reshape(-1, 1)).flatten()

    return df["close"].values[-60:], forecast_prices

if __name__ == "__main__":
    train_and_save_model()
