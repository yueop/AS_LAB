# 넘파이(NumPy)

## 개요

- ‘Numerical Python’의 줄임말로, 파이썬에서 벡터, 행렬 등의 다차원 배열을 효율적으로 처리할 수 있도록 도와주는 라이브러리.
- 고성능 수치 계산을 위해 설계, 큰 배열과 행렬 연산에 필요한 다양한 함수 제공.
- N-차원의 배열을 효과적으로 다룰 수 있음.

## 필요성 및 응용 분야

- 효율적인 데이터 저장과 접근(파이썬 리스트에 비해 더 적은 메모리 차지, 연속된 메모리 블록에 데이터 저장하여 배열 연산을 빠르게 처리하도록  도와줌).
- 벡터화 연산(반복문 없이 전체 배열에 대한 연산을 수행가능하여 코드 간결화).
- 다양한 수학 함수와 통계 연산 제공
- 과학계산 및 데이터 분석
- 즉, 고성능의 다차원 배열을 효과적으로 처리할 수 있도록 만들어진 라이브러리.

## 특징

- 데이터 형식: 동일한 데이터 형식으로만 구성됨
- 메모리 사용: 효율적임
- 연산 속도: 빠름
- 지원 함수: 풍부함
- 응용 분야: 과학 계산 및 데이터 분석

## 데이터 구조

- 스칼라: 한 개의 숫자로 이루어진 차원이 없는 데이터(Ex: 11)
- 벡터: 스칼라의 집합, 여러 개의 숫자가 순서대로 모여있는 1차원 데이터(Ex: [5, 3, 7] 행벡터 / [5, 1, 2] 열벡터) 크기와 방향을 가짐.

- 행렬: 벡터의 집합, 한 개 이상의 행과 열로 구성된 2차원 배열(데이터)

        (Ex: [[4, 19, 8],[16, 3, 5]])

- 텐서: 행렬의 집합, 3차원 이상의 배열(데이터)

        (Ex: [[[1, 2, 3][4, 5, 6]][[a, b, c][e, f, g]]])

## 데이터 형

- int: 정수(np,int8, np.int16, np.int32, np.int64)
- uint: 양의 정수(np.uint8, np.uint16, np.uint32, np.uint64)
- bool: 논리(bool)
- float: 실수(np.float16, np.float32, np.float64)
- complex: 복소수(np.complex64, np.complex128)
- 각 배열은 한 개의 데이터형만 가질 수 있다.
- dtype 메소드를 이용하여 데이터형을 파악하거나 배열 생성 시 타입을 미리 지정하여 생성 가능.

## 배열 생성

- np.array(): 리스트나 튜플로부터 배열 생성

```python
import numpy as np

#1차원 배열
arr_1d = np.array([1, 2, 3, 4, 5])

#2차원 배열
arr_2d = np.array([1, 2, 3][4, 5, 6])
```

- np.arange(): 일정한 간격의 값들을 가진 배열 생성

```python
import numpy as np

#0부터 9까지
arr = np.arange(10)

#2부터 19까지, 간격은 2
arr = np.arange(2, 20, 2)
```

- np.zeros(): 모든 값이 0인 배열 생성

```python
import numpy as np

#1차원 배열
zeros_1d = np.zeros(5)

#2차원 배열
zeros_2d = np.zeros((3, 4))
```

- np.ones(): 모든 값이 1인 배열 생성

```python
import numpy as np

#1차원 배열
ones_1d = np.ones(5)

#2차원 배열
ones_2d = np.ones((3, 4))
```

- 기본적으로 실수형으로 생성되기 때문에 dtype을 int로 지정해야 정수형으로 생성

Ex) zeros_1d = np.zeros(5, dtype=int)

- np.random.rand(): 무작위 값들을 가진 배열 생성(0에서 1 사이의 값)

```python
import numpy as np

#1차원 배열
random_arr_1d = np.random.rand(5)

#2차원 배열
random_arr_2d = np.random.rand(3, 4)
```

- np.linspace(): 지정한 범위 내에서 균일한 간격의 값들을 가진 배열 생성

```python
import numpy as np

#0부터 1까지 5개의 값
arr = np.linspace(0, 1, 5)
```

- np.linspace(start, stop, num, endpoint=True, retstep=False, dtype-None, axis=0)
- num: 생성할 샘플의 개수
- endpoint: True로 설정되면 stop값이 샘플의 마지막이 되고, False면 stop값은 배열에 포함되지 않는다.(디폴트 값 True)
- retstep: True로 설정되면 spmple, step의 튜플 반환.(디폴트 값 False)
- axis: 반환되는 배열의 축(디폴트 값 0)

## 연산

- 다양한 수학, 통계, 공학 함수 제공

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("합: ", np.sum(arr))
print("행의 원소 합: ", np.sum(arr, axis=0))
print("열의 원소 합: ", np.sum(arr, axis=1))
print("평균: ", np.mean(arr))
print("행의 원소 평균: ", np.mean(arr, axis=0))
print("열의 원소 평균: ", np.mean(arr, axis=1))
print("최대: ", np.max(arr))
print("최소: ", np.min(arr))
print("행의 원소 최대값: ", np.max(arr, axis=0))
print("행의 원소 최소값: ", np.min(arr, axis=0))
print("열의 원소 최대값: ", np.max(arr, axis=1))
print("열의 원소 최소값: ", np.min(arr, axis=1))
```

- 인덱스의 최대, 최소값은 np.argmax(), np.argmin() 함수로 찾는다.
- 분산 및 표준편차는 np.var(), np.std() 함수로 구할 수 있다.

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

print("분산: ", np.var(arr))
print("표준편차: ", np.std(arr))
```

- 배열의 속성 출력

```python
import numpy as np

arr1 = np.ones([3, 4])

print(arr1)
print('size: ', arr1.size)
print('dtype: ', arr1.dtype)
print('shape: ', arr1.shape)
print('ndim: ', arr1.ndim)
```

- 배열의 인덱싱과 슬라이싱:  데이터 추출 시 가장 기본이 되는 연산

```
import numpy as np

arr1 = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print(arr1[0])
print(arr1[0, 2])
print(arr1[1:])
print(arr1[arr1>5])
```

- 배열의 변형: transpose(), flatten(), reshape()

```python
import numpy as np

arr = np.array([[1, 2, 3],[4, 5, 6]])

arr_T=arr.transpose()
arr_F=arr.flatten()
arr_R=arr.reshape(3, 2)
print(arr)
print(arr_T)
print(arr_F)
print(arr_R)
```

- 연산자에 의한 기본 연산

```python
import numpy as np

arr1 = np.arange(1, 5)
arr2 = np.ones(4)

print(arr1 , arr2)
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 > arr2)
```

```python
import numpy as np

arr1 = np.random.randint(11, size=4)
print(arr1)
print(arr1 + 2)
print(arr1 * 2)
```

- 브로드캐스팅(broadcasting)
일정 조건에 맞으면 서로 다른 모양의 배열끼리 연산할 수 있도록 하는 기능(조건: 한 배열의 차원이 1일 때, 첫 번째 배열의 행의 개수와 두 번째 배열의 열의 개수가 같을 때)
