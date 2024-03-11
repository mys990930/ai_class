from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
df = pd.read_csv("./csvs/breast_cancer.csv", index_col=0)

X = df.drop('label',axis=1)
X.head()
y = df['label']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)
#x = 학습시킬 데이터, y = 답지
#train에 포함된 데이터로 train 시킨 모델을 사전에 분리해둔 test 값들과 비교해서 모델의 정확도 찾기

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