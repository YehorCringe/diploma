from flask import Flask, render_template, send_file
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    # Завантаження та підготовка даних
    model = load_model("models/test_model_gru_features_1d.keras")

    # Завантаження останніх даних з Binance (або з локального файлу)
    df = pd.read_csv("data/latest_data.csv")  # або API
    features = ["close", "SMA_10", "SMA_30", "EMA_10", "EMA_30", "RSI", "MACD", "Signal_Line", "volume"]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    def create_sequences(data, window_size):
        X = []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size, :])
        return np.array(X)

    window_size = 60
    X_input = create_sequences(scaled_data, window_size)
    predicted = model.predict(X_input)

    # Зворотне масштабування
    dummy = np.zeros((predicted.shape[0], scaled_data.shape[1] - 1))
    full_input = np.hstack((predicted, dummy))
    predicted_real = scaler.inverse_transform(full_input)[:, 0]

    # Побудова графіка
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_real, label="Прогнозовані ціни", color='red')
    plt.title("Прогнозована ціна Bitcoin")
    plt.xlabel("Час")
    plt.ylabel("Ціна, USDT")
    plt.legend()
    plt.tight_layout()

    # Зберігаємо графік
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig("static/prediction_plot.png")
    plt.close()

    return render_template("index.html", image_path="/static/prediction_plot.png")


if __name__ == "__main__":
    app.run(debug=True)
