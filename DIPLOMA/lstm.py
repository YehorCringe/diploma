# import api_test
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#
# # Створюємо модель LSTM
# model = Sequential([
#     LSTM(50, return_sequences=True, input_shape=(30, 1)),  # 50 нейронів, повертає послідовності
#     LSTM(50, return_sequences=False),  # Ще один LSTM-шар
#     Dense(25, activation="relu"),  # Повнозв'язний шар
#     Dense(1)  # Вихідний шар (прогноз ціни)
# ])
#
# # Компільовуємо модель
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# # Навчаємо модель
# history = model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_data=(X_test, Y_test))
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
