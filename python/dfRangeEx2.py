import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

data_loc = df.loc[2, ['A']]
print(data_loc)
data_iloc1 = df.iloc[1, 0]
print(data_iloc1)
data_iloc2 = df.iloc[0:3, 0:2]
print(data_iloc2)