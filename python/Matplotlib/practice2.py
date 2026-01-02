import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

penguins_data = pd.read_csv('penguins.csv')
sns.pairplot(penguins_data)
plt.show()