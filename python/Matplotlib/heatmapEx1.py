import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
df = iris_data.drop(['species'], axis=1)
sns.heatmap(data=df.corr(), annot=True, fmt='.2f', cbar=False)
plt.show()