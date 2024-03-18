from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
TF_ENABLE_ONEDNN_OPTS=0

# 데이터---------------------------------------------------------------------
df = pd.read_csv("./csvs/diabetes.csv", index_col=0)

X = df.drop('Outcome',axis=1)
X.head()
y = df['Outcome']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)
#x = 학습시킬 데이터, y = 답지
#train에 포함된 데이터로 train 시킨 모델을 사전에 분리해둔 test 값들과 비교해서 모델의 정확도 찾기

print(df['Outcome'].value_counts()) #레이블 확인

sns.countplot(data=df, x="Outcome") #그래프로 확인
plt.xlabel("Diabetes or not")
plt.ylabel("Count")
plt.title("Label")
plt.show()

print(df.isnull().sum()) #결측치 확인
#결측치가 없으므로 추가 처리 x

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

scaled_data = scaler_standard.fit_transform(df)
print("\n--- standard scaler 사용 데이터 ---")
print(scaled_data)

scaled_data = scaler_minmax.fit_transform(df)
print("\n--- minmax scaler 사용 데이터 ---")
print(scaled_data)

print("\n--- data 산점도 생성 ---")
plt.scatter(df['Outcome'], df['Age'])
plt.xlabel('has diabetes')
plt.ylabel('age')
plt.title('Diabetes per age')
plt.show()

print("\n--- box plot 생성 ---")
sns.boxplot(x="Age", data=df)
plt.title("Age ditribution")
plt.show()


# LR---------------------------------------------------------------------
clf_lr = LogisticRegression(random_state=0, max_iter=500)
clf_lr.fit(X_train, y_train)

pred_lr = clf_lr.predict(X_test) #X_test 내용으로 모델 만들기

print("\n--- Logistic Regression Classifier ---")
print(accuracy_score(y_test, pred_lr)) #만든 모델을 y_test와 비교하기
print(confusion_matrix(y_test, pred_lr))

# DT---------------------------------------------------------------------
clf_dt = DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train, y_train)

pred_dt = clf_dt.predict(X_test)

print("\n--- Decision Tree Classifier ---")
print(accuracy_score(y_test, pred_dt))
print(confusion_matrix(y_test, pred_dt))

# SVC---------------------------------------------------------------------
clf_svm = SVC(random_state=0)
clf_svm.fit(X_train, y_train)

pred_svm = clf_svm.predict(X_test)

print("\n--- SVM Classifier ---")
print(accuracy_score(y_test, pred_svm))
print(confusion_matrix(y_test, pred_svm))

# RF---------------------------------------------------------------------
warnings.filterwarnings('ignore')

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)

print ("\n--- Random Forest ---")
print(accuracy_score(y_test,pred))
print (confusion_matrix(y_test, pred))

