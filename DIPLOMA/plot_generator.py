import numpy as np
import matplotlib.pyplot as plt
from keras.src.saving import load_model
from api import X_test, features, scaler_close, prices


model = load_model("./model/GRUCNNrecursive.keras")
predicted_scaled = model.predict(X_test)

forecast_padded = np.zeros((len(predicted_scaled.flatten()), len(features)))
forecast_padded[:, 0] = predicted_scaled.flatten()

last_prediction = predicted_scaled[-1]
forecast_actual = scaler_close.inverse_transform(last_prediction.reshape(-1, 1)).flatten()

last_real_price = prices[-1]
delta = last_real_price - forecast_actual[0]
forecast_aligned = forecast_actual + delta


