import numpy as np
import matplotlib.pyplot as plt

preturi = [50000, 70000, 100000, 120000, 121000]
mpatr = [40, 50, 60, 70, 80]

n = len(preturi)

s_x = sum(mpatr)
s_y = sum(preturi)
s_xy = sum(mpatr[i] * preturi[i] for i in range(n))
s_x2 = sum(x**2 for x in mpatr)

m = (n * s_xy - s_x * s_y) / (n * s_x2 - s_x**2)
b = (s_y - m * s_x) / n


nou = 90
p = m * nou + b

print(p)

plt.scatter(mpatr, preturi, color='blue', label='preturi')
plt.plot(mpatr, [m * x + b for x in mpatr], color='red', label='Regresie')


plt.show()