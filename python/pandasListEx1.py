import pandas as pd

data = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']] #2차원 리스트 생성
df1 = pd.DataFrame(data, columns=['ID', 'Name']) #2차원 리스트를 데이터프레임으로 변환
print(df1)