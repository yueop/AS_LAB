import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 11)
y = np.random.randint(1, 30, 10)

plt.scatter(x, y, c='r', marker='s')
plt.show()