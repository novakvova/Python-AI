import numpy as np;
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt

#Попит на морозиво
temperature = np.array([2,5,8,12,15,18,20,22,25,28,30,33])
ice_cream_sales = np.array([15,22,35,60,90,130,160,20,230,250,243,210])

print("Температура: ", temperature)
print("Кількість порцій морозива: ", ice_cream_sales)

X = temperature.reshape(-1,1) # 2D масив ознак
y = ice_cream_sales
# print("X", X)

#Будуємо модель для регресії полінома і даємо степінь
model = make_pipeline(PolynomialFeatures(5), LinearRegression())
model.fit(X, y) # передаємо значення у модель для навчання

x_smooth = np.linspace(temperature.min(),temperature.max(),200).reshape(-1,1)
y_smooth = model.predict(x_smooth) # Будемо робити прогнозування ціни для наших даних

new_temp = np.array([[27]])
predicted = model.predict(new_temp) # Прогноз для температури в 27 градусів


print(f"Прогноз для температури 27 C: {predicted[0]:.0f} шт")
print(f"R^2 моделі = {model.score(X, y):.3f}")

plt.scatter(temperature,ice_cream_sales, color="orange", label = "Реальні продажі")
plt.plot(x_smooth,y_smooth, color="navy", label = "Понілом 2-го ступеня")
plt.xlabel("Температура")
plt.ylabel("Продажі шт")
plt.legend()
plt.show()