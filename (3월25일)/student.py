import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./csvs/student-mat.csv", sep=';', index_col=0)

X = df[["Medu", "Fedu", "absences", "G1", "G2"]] #feature 2개 select
X.head()
y = df['G3']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

# LR---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print("\nfeatures: Medu, Fedu, absences, G1, G2")
print('평균제곱근오차', mean_squared_error(ly_preds, y_test))


#feature 종류 별 비교--------------------------------------------------
X = df[["failures", "studytime", "absences", "G1", "G2"]]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print("\nfeatures: failures, studytime, absences, G1, G2")
print('평균제곱근오차', mean_squared_error(ly_preds, y_test))
print("comment: 부모의 교육 수준보다 성적에 조금 더 직접적인 연관을 가진 feature들을 선택했을 때 정확도가 소폭 올라간 것을 확인할 수 있음")


X = df[["freetime", "goout", "health"]]
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print("\nfeatures: freetime, goout, health")
print('평균제곱근오차', mean_squared_error(ly_preds, y_test))
print("comment: 이전 시험 점수를 제외하고 간접적인 feature들만 선택했을 경우 정확도가 낮아지는 것을 확인할 수 있음")
