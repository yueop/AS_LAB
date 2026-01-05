import pandas as pd
import matplotlib.pyplot as plt

iris_data = pd.read_csv('iris.csv')
print(iris_data.describe())    #기초통계량 확인

iris_data.sort_values(by=['petal_length'], ascending=False) #petal_length 기준 내림차순 정렬

plt.scatter(x=iris_data['petal_length'], y=iris_data['petal_width'])    #산점도 그리기
plt.show()