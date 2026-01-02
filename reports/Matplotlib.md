# 맷플롯립(Matplotlib)

## 개요

- 주어진 데이터를 이용하여 다양한 형태의 그래프를 그리는 함수를 제공하는 파이썬 라이브러리
- 데이터를 그래프로 표현하면 데이터의 분포나 항목 간의 상관관계를 쉽게 파악할 수 있음
- math + plot + library의 합성어

## 맷플롯립으로 그래프 그리기

- 선 그래프: plot()

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [1, 4, 9, 16]
plt.plot(x, y)
plt.show()
```

-장점: 시간에 따른 변화를 한눈에 알 수 있다.

-plot(x값, y값, 선스타일문자열, marker=’마커기호’, label=’그래프이름’): y값은 필수이고 나머지는 생략 가능

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('data1.xlsx')

plt.plot(df['kor'], 'k--', marker='o')
plt.show()
```

-’k- -’: 검은색 점선

-marker=’o’: 값의 표시를 동그라미 마커로 표시

-그래프 및 옵션 추가

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('data1.xlsx')

plt.plot(df['kor'], 'k--', marker='o', label='korean')
plt.plot(df['math'], marker='^', label='math')
plt.show()
```

- 범례 추가: plt.legend()

- 산점도 그래프: scatter() 사용, 두 변수간의 관계 등 표현 시 사용

```python
import matplotlib.pyplot as plt

x = [6, 2, 1, 1, 4, 4, 6, 4, 7, 7]
y = [5, 4, 2, 9, 9, 0, 10, 3, 1, 2]

plt.scatter(x, y)
plt.show()
```

- Ex)

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 11)
y = np.random.randint(1, 30, 10)

plt.scatter(x, y, c='r', marker='s') #s는 square라는 뜻
plt.show()
```

-plt.scatter(x값, y값, c=’색 문자’ 혹은 ‘색깔 문자열’, marker=’마커기호’, labeel=’그래프 이름’): c, marker, label 생략 가능

- 막대 그래프: bar()/barh() 사용

```python
import matplotlib.pyplot as plt

x = ['A', 'B', 'C']
y = [100, 150, 120]

plt.bar(x, y)
plt.show()
```

-bar(x값, y값, color=’색 문자’ 혹은 ‘색깔 문자열’): 범주별 데이터를 나타내는 세로 막대 그래프 생성

-barh(x값, y값, color=’색 문자’ 혹은 ‘색깔 문자열’): 범주별 데이터를 나타내는 가로 막대 그래프 생성

- 빈도 그래프(히스토그램): hist() 사용, 데이터의 분포를 시각적으로 표현하는데 사용

```python
import matplotlib.pyplot as plt

x = [1, 1, 1, 2, 3, 2, 4]
plt.hist(x, bins=4, color='green', align='mid')
plt.show()
```

-plt.hist(x값, bins=x값 구간 개수(데이터의 두께), color=’색 문자’ 혹은 ‘색깔 문자열’, align=’left’/’mid’/’right’ 중 하나): bins에는 데이터의 구간 개수를 할당, 별도 지정하지 않으면 10을 할당

-align은 막대가 x축 눈금 대표값을 기준으로 왼쪽/가운데/오른쪽으로 저렬하도록 지정

- 그래프 제목, x축 제목, y축 제목 지정: title(), xlabel(), ylabel() 사용

```python
import matplotlib.pyplot as plt

x = ['A', 'B', 'C']
y = [100, 150, 120]
plt.bar(x, y)
plt.title('Test Graph')
plt.xlabel('Group')
plt.ylabel('Score')
plt.show()
```

- 한 화면에 여러 개 그래프 나타내기: subplots() 사용

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(x, y)
axs[0, 1].scatter(x, y)
axs[1, 0].bar(x, y)
axs[1, 1].barh(x, y)
fig.suptitle('Drawing Graph')
plt.show()
```

-subplots(행 개수, 열 개수, figsize=(가로 길이, 세로 길이)

## 맷플로립 사용시 한글 출력하는 방법

- 내컴퓨터에 설치된 한글 폰트들 중에서 하나를 선택하여 그래프를 그리는 데 사용하는 한글 폰트로 지정

```python
import matplotlib.pyplot as plt

plt.rc('font', family='Gulim')
x = [1, 2, 3, 4]
y = [10, 4, 15, 9]
plt.plot(x, y, label='국어')
plt.title('그래프 예제')
plt.xlabel('번호')
plt.ylabel('점수')
plt.legend()
plt.show()
```

- Ex)

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('data1.xlsx')
plt.rc('font', family='Gulim')
plt.plot(df['name'], df['kor'], 'g--', marker='o', label='국어점수')
plt.plot(df['name'], df['math'], 'r', marker='v', label='수학점수')
plt.title('성적 그래프')
plt.xlabel('이름')
plt.ylabel('점수')
plt.legend()
plt.show()
```

## 시본(Seaborn)

- 맷플롯립을 바탕으로 통계 그래프를 그릴 수 있도록 기능을 추가한 파이썬 라이브러리
- 판다스의 데이터프레임을 대상으로 그래프를 그리기 때문에 데이터프레임을 다양한 그래프로 나타내기가 편리함
- 타이타닉(titanic), 붓꽃(iris), 팁(tips), 여객운송(flights), 펭귄(penguins) 등의 샘플 데이터 셋을 제공

## 시본으로 그래프그리기

- 붓꽃 예제

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris_data)
plt.title('Scatter Plot by seaborn', fontsize=20)
plt.show()
```

-scatterplot(x=’x축에 표시할 칼럼명’, y=’y축에 표시할 칼럼명’, hue=’데이터 구분 칼럼명’, data=데이터프레임): hue값 지정은 생략 가능

- 범주별 데이터 개수를 표시하는 막대 그래프: countplot()사용

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.countplot(x='species', data=iris_data)
plt.show()
```

-countplot(x=’x축에 범주로 표시할 칼럼명’, hue=’데이터 구분 칼럼명’, data=데이터프레임): hue값 지정은 생략 가능

- 범주별 항목값 평균을 표시하는 막대 그래프: barplot()사용

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.barplot(x='species', y='petal_length', data=iris_data, errorbar='sd')
plt.show()
```

-barplot(x=’x축에 표시할 칼럼명’, y=’y축에 표시할 칼럼명’, hue=’데이터 구분 칼럼명’, data=데이터프레임)

-그래프 중앙의 수직선은 표준 편차를 나타내는 오차막대

- 숫자값을 갖는 칼럼들의 상관관계를 보여주는 그래프: pairplot()사용

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.pairplot(iris_data, hue='species')
plt.show()
```

-pairplot(데이터프레임, hue=’범주를 갖는 칼럼명’)

- 산점도와 추세선(회귀선)을 동시에 보여주는 그래프: regplot()사용

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
sns.regplot(x='petal_length', y='petal_width', data=iris_data)
plt.show()
```

-regplot(x=’x축에 표시할 칼럼명’, y=’y축에 표시할 칼럼명’, data=데이터프레임)

-회귀선은 예측할 때 선형회귀의 회귀선을 의미한다ㅏ.

- 집계한 값에 따라 색깔을 표시하는 그래프: heatmap()사용

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('iris.csv')
df = iris_data.drop(['species'], axis=1)
sns.heatmap(data=df.corr(), annot=True, fmt='.2f', cbar=False)
plt.show()
```

-hearmap(data=2차원 행렬 형태로 전달된 데이터, annot=True/False, fmt=’파이썬 서식 문자열’, cbar=True/False): annot: True면 숫자 표현, cmap: 색상 지정

-선형적 강도와 방향을 나타내면서 -1부터 +1까지의 값을 가진다.

-점수가 높을수록 두 변수간의 상관관계가 높다는 것을 의미한다.
