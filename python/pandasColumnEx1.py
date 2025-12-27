import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

#단일 열 선택 예제
column_a1 = df.A    #속성 접근 방식
column_a2 = df['A'] #딕셔너리 접근 방식

print(column_a1)
print(column_a2)

#여러 열 선택 예제
columns_ab = df[['A', 'B']]
print(columns_ab)