split: 문자열을 특정 구분자로 나누어 리스트로 반환하는 메소드.

Ex) txt = “apple#banana#cherry#orange”

x = txt.split(”#”)

print(x)

실행결과: [’apple’, ‘banana’, ‘cherry’, ‘orange’]

*구분자 생략시 공백에서 분할

join: 리스트의 각 요소를 문자열로 변환하고, 주어진 구분자로 결합하여 하나의 문자열로 반환

Ex) words = [’Hello’, ‘world’, ‘this’, ‘is’, ‘a’, ‘test’]

sentence = ‘ ‘.join(words)

print(sentence)

실행결과: Hello world this is a test

list comprehension: 리스트를 쉽고 짧게 한 줄로 만들 수 있는 파이썬 문법.

- 형식: newlist = [expression for item in iterable if condition]
- 반환값은 새로운 리스트, 기존 리스트는 변경되지 않는다.
- new_list: 새로운 리스트를 저장할 변수 이름.
- expression: 각 요소에 대한 계산식 또는 변환식
- item: 반복 가능한(iterable) 객체에서 가져올 각 요소의 변수 이름.
- iterable: 반복 가능한 객체(예: 리스트, 튜플, 문자열 등).
- condition(생략가능): 조건식. 이 조건식을 만족하는 요소만 새 리스트에 포함.

Ex) number_list = [x**2 for x in range(10) if x % 2 == 0]

print(number_list)

실행결과: [0, 2, 4, 6, 8]

enumerate: 순회 가능한 객체를 인덱스와 함께 반환

Ex)fruits = [’apple’, ‘banana’, ‘cherry’]

for idx, fruit in enumerate(fruits):

print(idx, fruits)

실행결과: 0 apple

1 banana

2 cherry

zip: 두 개 이상의 순회 가능한 객체를 병렬적으로 순회하면서 각 항목을 튜플로 묶어 반환.

Ex)zip(iterator1, iterator2, …)

lambda: 이름 없는 함수(익명 함수)를 정의할 때 사용(간단한 함수를 간결하게 표현할 때 유용)

Ex)add = lambda x, y: x + y

```python
words = ["apple", "banana", "cherry", "date", "elderberry"]
words.sort(key = lambda word: len(word))

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
```

map: 함수와 순회 가능한 객체를 받아, 객체의 각 항목에 함수를 적용한 결과를 반환.

```python
def convert_to_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'
    
scores = [88, 92, 78, 60, 75, 95]
grades = list(map(convert_to_grade, scores))
```

filter: 함수와 순회 가능한 객체를 받아, 넘어온 조건 함수를 만족하는 데이터만 찾아서 반환.
