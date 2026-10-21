import numpy as np
import matplotlib.pyplot as plt
from fontTools.merge import layout

x = np.linspace(0,2,100)

# figsize - розмір в дюймах
# constrained - відступи будуть підбиратися автоматично
fig, ax = plt.subplots(figsize=(5,2.7), layout='constrained')
ax.plot(x, x, label='лінія')
ax.plot(x, x**2, label = "квадратна")
ax.plot(x, x**3, label = "кубічна")

ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title("Лінійні графіки")
ax.legend() # Додаємо легенду, яку задали через label
plt.show() # показуємо графік