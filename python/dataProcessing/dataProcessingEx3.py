import pandas as pd

df = pd.read_csv('iris.csv')
print(df[df.duplicated()])

df2 = df.drop_duplicates()
print(df2)