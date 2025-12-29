# 판다스(Pandas)

## 개요

- 데이터 분석을 위해 많이 사용하는 라이브러리
- 넘파이를 기반으로 만들어졌기 때문에 대용량 데이터를 빠른 속도로 분석 가능
- 기본 데이터 구조로 1차원 배열 형태인 시리즈(Series)와 2차원 배열 형태인 데이터프레임(DataFrame)이 있음

## 데이터 프레임의 구조

1. 레코드: 행(row)을 나타냄. 특정 개체에 대한 정보 포함.
2. 칼럼: 열(column)을 나타냄. 같은 타입의 데이터를 담고, 특정 속성이나 특성을 나타냄.
3. 행 번호(index): 각각의 레코드의 식별하는데 사용.
4. 컬럼명: 각 컬럼을 식별하는데 사용하는 레이블. 데이터의 속성이나 특성을 나타냄. 데이터 프레임의 컬럼을 참조할 때 사용.

## 특징

- 데이터 분석과 처리 작업을 빠르고 효율적으로 수행할 수 있다.
- 구조가 굉장히 유연하고 사용자의 구조에 따라 쉽게 변경가능하다.

Ex) -데이터프레임에 칼럼과 행을 쉽게 추가하거나 삭제하거나 순서 변경도 가능

-데이터프레임의 각 칼럼에 다른 타입의 데이터를 저장 가능

-처음에 지정한 칼럼의 데이터 타입 변경 가능

-데이터프레임의 일부를 분리하거나 병합할 수 있음

- 데이터 처리를 쉽게 할 수 있다.

Ex) -누락된 데이터를 쉽게 처리하기 위한 기능들 제공

-데이터의 집계와 변환 작업을 위해 특정 칼럼을 기준으로 그룹화하는 기능 제공

-슬라이싱과 조건식을 활용해 원하는 데이터만 쉽게 가져올 수 있음

-데이터 정렬, 통계 결과 산출 용이

-날짜 데이터에서 연, 월, 일, 요일을 추출하는 등의 다양한 함수 제공(시계열 데이터 분석에 유용)

## DataFrame메소드를 이용하여 데이터프레임 생성

- 파이썬 리스트를 데이터프레임으로 생성

```python
import pandas as pd

data = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']] #2차원 리스트 생성
df1 = pd.DataFrame(data, columns=['ID', 'Name']) #2차원 리스트를 데이터프레임으로 변환
print(df1)
```

- index옵션을 사용하여 인덱스 명 지정

```python
import pandas as pd

data = [[1, 'Alice'], [2, 'Bob'], [3, 'Charlie']] #2차원 리스트 생성
df1 = pd.DataFrame(data, columns=['ID', 'Name'], index=['a', 'b', 'c']) #2차원 리스트를 데이터프레임으로 변환
print(df1)
```

- 파이썬 딕셔너리를 데이터프레임으로 생성

```python
import pandas as pd

df = pd.DataFrame({'name':['Kim','Lee','Park',],
                   'age':[27,33,19],
                   'score':[92,98,87]})
print(df)
```

## 외부 파일을 읽어 데이터프레임 생성

- CSV(Comma-Separated Values) 파일 읽기: read_csv(’csv파일명’, encoding=’인코딩 타입’, header=None, index_col=0)
- header=None: 제목행이 없는 경우에 지정하는 옵션
- index_col=0: 자동으로 붙는 행 인덱스 번호 대신 특별한 열을 인덱스로 사용할 때 지정하는 옵션

```python
import pandas as pd
import os

# 현재 파이썬 파일이 있는 폴더의 절대 경로를 계산
current_dir = os.path.dirname(os.path.abspath(__file__))

# 그 폴더 안에 있는 'data.csv'를 연결
file_path = os.path.join(current_dir, 'data.csv')
df = [pd.read](http://pd.read)_csv(file_path, encoding='UTF-8')
print(df)
```

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.csv')

df = [pd.read](http://pd.read)_csv(file_path, index_col=0)
print(df)
```

- 탭으로 구분된 파일 읽기

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv'
df = [pd.read](http://pd.read)_csv(url, delimiter='\t')
print(df)
```

## xlsx 파일을 읽어 데이터프레임 생성

- .read_excel(’xlsx 파일명’): xlsx파일을 읽어서 파일 데이터의 첫 번째 줄을 칼럼명으로 갖는 데이터프레임 생성

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)
print(df)
```

## 데이터프레임 데이터 보기

- 데이터프레임의 상위/하위 데이터 몇 개만 보기: head()/tail(): default값=5

```python
print(df.head(3))
print(df.tail(3))
```

- 데이터프레임에 대한 정보 보기: info(): 데이터프레임의 칼럼들과 데이터 타입, 데이터 개수 등의 기본 정보를 출력한다.

-object: 문자열을 의미

-NaN: 결측값(Missing Value)을 의미

- 데이터프레임의 특정 칼럼의 데이터 보기

```python
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
```

- 데이터프레임의 행 범위를 지정해서 데이터 보기

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

rows_1_to_3 = df[1:4]
print(rows_1_to_3)
column_a_rows_1_to_3 = df['A'][1:4]
print(column_a_rows_1_to_3)
```

- 데이터프레임에서 행과 칼럼을 지정하여 데이터 보기

-iloc: integer loc

```python
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
```

- 데이터프레임에 대한 조건 연산 결과 출력하기

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e']
})

condition_result = df['A'] > 3  #'A'열에서 값이 3보다 큰지 확인
print(condition_result)
```

- 데이터프레임의 데이터 정렬

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

df.sort_values(by=['이름'])
print(df)
```

-df.sort_values(by=[’칼럼명’], ascending=True/False, inplace=True/False)형식

-ascending은 True면 오름차순 정렬(디폴트는 True)

-inplace는 True면 원본 수정 False면 복사본 생성(디폴트는 False)

- 데이터프레임의 속성값 정보 보기

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

print(list(df.index))
print(list(df.columns))
print(df.size)
```

-df.index: 데이터프레임이 갖는 행 번호들

-df.columns: 데이터프레임이 갖는 칼럼명들

-df.size: 데이터프레임에 있는 개별 항목의 개수들(행의 수 * 열의 수)

## 데이터프레임의 데이터 집계

- 데이터프레임의 전체 기본 통계 출력

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

print(df.describe())
```

-df.describe(): 데이터프레임에서 숫자값을 갖는 칼럼들에 대한 기본 통계(개수, 평균, 표준편차, 최소값, 4분위 값)를 출력

- 칼럼값 별 데이터 개수 보기

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

print(df['반'].value_counts())
```

-df[’칼럼 이름’].value_counts(): 데이터프레임에서 지정한 칼럼의 값별로 데이터 개수 출력

- 개별 칼럼의 통계값 구하기

```python
import pandas as pd

data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10]
}

df = pd.DataFrame(data)

sum_A = df['A'].sum()   #칼럼 'A'의 합계 계산
print(f'sum of column A: {sum_A}')
```

-sum(), count(), mean(), std(), max(), min(), median() 등 통계함수를 이용해 원하는 통계값을 계산(std = 표준편차, median = 중간값)

- 그룹별 통계값 구하기

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

grouped_sum = df.groupby('반')['국어'].sum()
print(grouped_sum)

grouped_mean = df.groupby('반')['국어'].mean()
print(grouped_mean)

grouped_std = df.groupby('반')['국어'].std()
print(grouped_std)
```

-groupby()를 사용하여 그룹으로 묶을 칼럼을 지정한 뒤에 통계함수를 이용해서 지정한 칼럼별로 구분한 통계값을 계산

## 데이터프레임에서 결측치 처리

- 데이터프레임에서 결측치 확인

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

print(df.isna())
print(df.isna().sum())
```

-df.isna(): 칼럼의 값이 없으면 True, 값이 있으면 False로 출력

- 데이터프레임에서 결측치에 값 채우기

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

df2 = df.fillna({'확인여부':'완료'})    #데이터프레임(df)의 '확인여부' 컬럼의 결측치를 '완료'로 채움
print(df2)

df3 = df2.fillna(0)   #데이터프레임(df2)의 전체 칼럼에서 결측치를 0으로 채움
print(df3)
```

- 데이터프레임에서 결측치가 있는 데이터 삭제

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

df2 = df.dropna(subset=['응시여부']) #데이터프레임의 '응시여부' 칼럼에서 결측치가 있는 행 제거
print(df2)

df3 = df.dropna()
print(df3) #데이터프레임의 모든 칼럼에서 결측치가 있는 행 제거
```

-df.dropna(): axis=0이면 결측치가 있는 행을, 1이면 결측치가 있는 칼럼을 삭제

-subset으로 특정 칼럼을 지정하면 지정한 칼럼에 결측치가 있는 데이터를 삭제

## 데이터프레임의 변경

- 데이터프레임의 칼럼을 이용하여 새 칼럼 추가

```python
import pandas as pd
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(cur_dir, 'data.xlsx')
df = [pd.read](http://pd.read)_excel(file_path)

df['총점'] = df['국어'] + df['수학']
print(df)
```

- 데이터프레임의 행/칼럼 삭제

```python
import pandas as pd

df = pd.DataFrame({
    'Column1': [1, 2, 3, 4],
    'Column2': [5, 6, 7, 8]
})

df.drop(index=1, axis=0, inplace=True)  # 인덱스 1에 해당하는 행 삭제
print(df)
```

-drop(): 인덱스값으로 지정한 행(레코드)/칼럼명으로 지정한 칼럼을 데이터프레임에서 삭제

- 데이터프레임의 항목값의 일괄 변경

```python
import pandas as pd

df = pd.DataFrame({
    'Column1': [1, 2, 3, 4],
    'Column2': [5, 6, 7, 8]
})

#값 3을 300으로 변경
df.replace(3, 300, inplace=True)
print(df)
```

-df.replace(3, 300, inplace=True): 원본 데이터프레임에서 3을 모두 찾아 7로 바꾸기

-df.replace({1:10}, {2:20}): 원본 데이터프레임에서 1→10, 2→20으로 바꾸기

-df.replace({’단가’:{0:1000},

‘수량’:{0:100}}): 원본 데이터프레임에서 ‘단가’ 칼럼의 0은 1000으로, ‘수량’ 칼럼의 0은 100으로 바꾸기
