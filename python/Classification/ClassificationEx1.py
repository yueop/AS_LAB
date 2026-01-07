import pandas as pd
#아이리스 데이터
from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.model_selection import train_test_split    #데이터를 훈련 및 테스트 세트로 분할
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score #분류 모델의 정확도 평가

iris.keys() #iris 데이터의 key 확인해보기
iris.data #iris 데이터의 Feature(입력변수)
iris.target #iris 데이터의 Label(출력데이터)
iris.feature_names  #feature_names(입력변수 각 이름) 확인하기
iris.target_names   #target_names(예측하려는 값(class)을 가진 문자열 배열) 확인하기
print(iris.DESCR)   #데이터셋의 설명

#데이터프레임에 아이리스 데이터 담아주기
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.head()

#꽃잎 길이, 꽃잎 너비만 추출하여 iris_petal에 저장
iris_petal = iris.data[:, [2,3]]
print(iris_petal)

#데이터 크기 확인
iris_df.shape

#데이터프레임에 label 컬럼 추가하기
iris_df["label"] = iris.target
iris_df.head()

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, #input data(feature)
                                                    iris.target, #output data(label or target)
                                                    test_size = 0.2,
                                                    random_state= 7)
print('X_train 개수: ', len(X_train), 'X_test 개수: ', len(X_test))

#시각화
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 10)) #인치
sns.pairplot(iris_df, hue='label', palette='bright') #모든 변수 쌍에 대한 산점도 그리기
plt.show()

#결정트리 알고리즘 모델 생성
dt_model = DecisionTreeClassifier(random_state=32)
dt_model.fit(X_train, Y_train)  #학습하기

#예측 및 정확도 계산
y_pred = dt_model.predict(X_test)
print(y_pred)
print(Y_test)   #정답 비교

print("정확도: ", accuracy_score(Y_test, y_pred))

#모델별 성능 평가 지표
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))