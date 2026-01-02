import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

penguins_data = pd.read_csv('penguins.csv')
sns.barplot(x='species', y='body_mass_g', data=penguins_data, hue='sex')
plt.show()