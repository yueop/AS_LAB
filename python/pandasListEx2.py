import pandas as pd

data = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']] #2차원 리스트 생성
#index옵션 사용하여 인덱스 번호에 이름 부여
df1 = pd.DataFrame(data, columns=['ID', 'Name'], index=['a', 'b', 'c']) #2차원 리스트를 데이터프레임으로 변환
print(df1)