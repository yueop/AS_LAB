import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

rows_1_to_3 = df[1:4]
print(rows_1_to_3)
column_a_rows_1_to_3 = df['A'][1:4]
print(column_a_rows_1_to_3)