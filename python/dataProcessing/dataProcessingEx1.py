import pandas as pd
df = pd.read_csv('iris.csv')
df.info()

print(df.isnull().sum())
print(df.isna().sum())