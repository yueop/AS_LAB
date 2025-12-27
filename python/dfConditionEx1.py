import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

condition_result = df['A'] > 3  #'A'열에서 값이 3보다 큰지 확인
print(condition_result)