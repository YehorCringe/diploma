# import numpy as np
# import api_test
#
# # Прогнозуємо значення на тестовому наборі
# predicted_prices = api_test.model.predict(api_test.X_test)
#
# # Зворотна нормалізація (перетворюємо назад у реальні ціни)
# predicted_prices = api_test.scaler.inverse_transform(predicted_prices)
#
# # Також денормалізуємо реальні значення
# real_prices = api_test.scaler.inverse_transform(api_test.Y_test.reshape(-1, 1))
#
# # Візуалізація результатів
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 6))
# plt.plot(api_test.df["timestamp"].iloc[-len(real_prices):], real_prices, label="Реальні ціни", color="blue")
# plt.plot(api_test.df["timestamp"].iloc[-len(predicted_prices):], predicted_prices, label="Прогнозовані ціни", color="red")
# plt.xlabel("Дата")
# plt.ylabel("Ціна BTC/USDT")
# plt.title("Прогнозовані vs Реальні ціни Bitcoin")
# plt.legend()
# plt.grid(True)
# plt.show()
