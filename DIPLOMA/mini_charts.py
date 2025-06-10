import requests
import pandas as pd
import time

CACHE = {
    "binance": None,
    "bybit": None,
    "timestamp": 0
}
CACHE_TIMEOUT = 300  # 5 хвилин

def fetch_binance_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 30
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        prices = [float(entry[4]) for entry in data]  # Закриття
        return prices
    except Exception as e:
        return []

def fetch_bybit_data():
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": "BTCUSDT",
        "interval": "D",  # Daily
        "limit": 30
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        # Перевіримо, що API дало правильну відповідь
        if "result" in data and "list" in data["result"]:
            prices = [float(item[4]) for item in data["result"]["list"]][::-1]  # 4 — ціна закриття
            return prices
        else:
            print("Bybit: неправильна структура відповіді:", data)
            return []
    except Exception as e:
        print("Bybit API error:", e)
        return []


def get_mini_chart_data():
    now = time.time()
    if now - CACHE["timestamp"] > CACHE_TIMEOUT:
        CACHE["binance"] = fetch_binance_data()
        CACHE["bybit"] = fetch_bybit_data()
        CACHE["timestamp"] = now

    return {
        "binance": CACHE["binance"],
        "bybit": CACHE["bybit"]
    }
