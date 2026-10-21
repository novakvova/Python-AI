import matplotlib.pyplot as plt
import numpy as np

print("-- Працюємо із графіками --")

# набір величин - точок по х
x = np.linspace(0,2*np.pi,200)
# знаходимо sin від х
y = np.sin(x)
#Створюємо фігуру для малювання
fig, ax = plt.subplots() # Figure - усе вікно графіку
ax.plot(x,y) # - звичайний лінійний графік
plt.show() # відображаємо графік функції