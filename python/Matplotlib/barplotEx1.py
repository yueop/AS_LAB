import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.barplot(x='species', y='petal_length', data=iris_data, errorbar='sd')
plt.show()
