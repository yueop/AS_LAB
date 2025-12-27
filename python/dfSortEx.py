import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = pd.read_excel(file_path)

df.sort_values(by=['이름'])
print(df)