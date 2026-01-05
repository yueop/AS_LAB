import pandas as pd
import numpy as np

df = pd.read_excel('data.xlsx')
df.info()

print(df.isnull().sum())
print(df.isna().sum())

df.dropna(how='any')    #결측치가 하나라도 있는 행 삭제
df.fillna(0, inplace=True)  #결측치를 0으로 채우기
df.replace(np.nan, 0)  #결측치를 0으로 대체