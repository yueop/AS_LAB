import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.countplot(x='species', data=iris_data)
plt.show()