import matplotlib.pyplot as plt

x = [1, 1, 1, 2, 3, 2, 4]
plt.hist(x, bins=4, color='green', align='mid')
plt.show()