from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

# ─── Для роботи з моделями і прогнозом ────────────────────────────────────────
from keras.src.saving import load_model
from train_model import (
    fetch_binance_data as fetch_binance_df,   # отримує DataFrame від Binance
    compute_indicators,
    run_forecast_pipeline
)
# ──────────────────────────────────────────────────────────────────────────────

# ─── Для міні-графіків поточних цін ────────────────────────────────────────────
from mini_charts import get_mini_chart_data  # функція повертає dict із 30 днями для Binance і Bybit
# ──────────────────────────────────────────────────────────────────────────────

# ─── Для відображення на головній сторінці ────────────────────────────────────
from api import prices, get_btc_prices_from_exchanges, X_test, features, scaler_close
from plot_generator import forecast_aligned
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = 'appdiploma'
app.db_initialized = False  # прапорець для ініціалізації БД

# ─── Таймер (якщо потрібно використовувати APScheduler) ─────────────────────────
scheduler = BackgroundScheduler()
# scheduler.add_job(generate_forecast_plot, 'interval', hours=1)
# ──────────────────────────────────────────────────────────────────────────────

# ─────────────── Ініціалізація бази даних ─────────────────────────────────────
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.before_request
def initialize_db_once():
    if not app.db_initialized:
        init_db()
        app.db_initialized = True
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────── Реєстрація користувача ──────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (email, password) VALUES (?, ?)',
                (email, password)
            )
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "Пошта вже зайнята!"
    return render_template('register.html')
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────── Аутентифікація ───────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT password FROM users WHERE email = ?',
            (email,)
        )
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['email'] = email
            return redirect(url_for('index'))
        else:
            return "Невірна пошта або пароль!"
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────── Головна сторінка (index) ─────────────────────────────────
@app.route('/')
def index():
    # якщо не залогінений, перенаправити на сторінку /login
    if 'email' not in session:
        return redirect(url_for('login'))

    # Отримуємо дві змінні для початкового (статичного) відображення:
    # real_prices      – це останні 60 справжніх значень з api.py → prices
    # forecast_prices  – це align-прогноз, розрахований у plot_generator.py
    btc_prices = get_btc_prices_from_exchanges()
    real_prices = prices[-60:].tolist()
    forecast_prices = forecast_aligned.tolist()

    return render_template(
        'index.html',
        real_prices=real_prices,
        forecast_prices=forecast_prices,
        btc_prices=btc_prices
    )
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────── Ендпойнт міні-графіків ───────────────────────────────────
@app.route('/api/mini-charts')
def mini_charts():
    data = get_mini_chart_data()  # повертає {'binance': [...30 close...], 'bybit': [...30 close...]}
    return jsonify(data)
# ──────────────────────────────────────────────────────────────────────────────


# ─────────────────── Ендпойнт прогнозу (через сегмент /api/forecast/<exchange>) ─
@app.route("/api/forecast/<exchange>")
def forecast_api(exchange):
    """
    Цей маршрут повертає JSON із полями:
      - real: останні 60 реальних цін
      - forecast: масив прогнозу (7 точок)
      - indicators: словник, де кожен ключ — це масив останніх 60 точок відповідного індикатора
    """
    # Обираємо DataFrame і модель залежно від назви біржі
    if exchange == "binance":
        df = fetch_binance_df()  # повертає DataFrame з Binance (вся історія)
        model = load_model("models/model_binance.keras")
    else:
        return jsonify({"error": "Unsupported exchange"}), 400

    # 1) Обчислюємо усі індикатори на повному DataFrame (функція повертає DataFrame із новими стовпцями):
    df_ind = compute_indicators(df.copy())

    # 2) Отримуємо last_60 real і прогноз через run_forecast_pipeline:
    real_prices_all, forecast_prices = run_forecast_pipeline(df.copy(), model)
    real_60 = real_prices_all[-60:].tolist()

    # 3) Для кожного індикатора відбираємо останні 60 значень (dropna(), щоб усі NaN-и прибрати, якщо вони є на початку)
    indic = {
        "SMA_10":      df_ind["SMA_10"].dropna().tolist()[-60:],
        "SMA_30":      df_ind["SMA_30"].dropna().tolist()[-60:],
        "EMA_10":      df_ind["EMA_10"].dropna().tolist()[-60:],
        "EMA_30":      df_ind["EMA_30"].dropna().tolist()[-60:],
        "RSI":         df_ind["RSI"].dropna().tolist()[-60:],
        "MACD":        df_ind["MACD"].dropna().tolist()[-60:],
        "Signal_Line": df_ind["Signal_Line"].dropna().tolist()[-60:]
    }

    return jsonify({
        "real":       real_60,
        "forecast":   forecast_prices.tolist(),
        "indicators": indic
    })


# ─────────────────── Ендпойнт прогнозу (через query-param /api/forecast?exchange=...) ─
@app.route("/api/forecast")
def get_forecast():
    """
    Аналогічний маршрут, але використовує ?exchange=…
    """
    exchange = request.args.get("exchange", "binance")

    if exchange == "binance":
        df = fetch_binance_df()
        model = load_model("models/model_binance.keras")
    else:
        return jsonify({"error": "Невідома біржа"}), 400

    df_ind = compute_indicators(df.copy())
    real_prices_all, forecast_prices = run_forecast_pipeline(df.copy(), model)
    real_60 = real_prices_all[-60:].tolist()

    indic = {
        "SMA_10":      df_ind["SMA_10"].dropna().tolist()[-60:],
        "SMA_30":      df_ind["SMA_30"].dropna().tolist()[-60:],
        "EMA_10":      df_ind["EMA_10"].dropna().tolist()[-60:],
        "EMA_30":      df_ind["EMA_30"].dropna().tolist()[-60:],
        "RSI":         df_ind["RSI"].dropna().tolist()[-60:],
        "MACD":        df_ind["MACD"].dropna().tolist()[-60:],
        "Signal_Line": df_ind["Signal_Line"].dropna().tolist()[-60:]
    }

    return jsonify({
        "real":       real_60,
        "forecast":   forecast_prices.tolist(),
        "indicators": indic
    })


# ─────────────────── Запуск Flask ─────────────────────────────────────────────
if __name__ == '__main__':
    scheduler.start()
    app.run(debug=True)
