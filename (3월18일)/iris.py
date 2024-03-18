import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

TF_ENABLE_ONEDNN_OPTS = 0
columns_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("./csvs/iris.data", header=None, names=columns_name)

X = df.drop('class',axis=1)
X.head()
y = df['class']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)

print("\nX_train: ")
print(X_train)
print("\nX_test: ")
print(X_test)
print("\nY_train: ")
print(y_train)
print("\nY_test: ")
print(y_test)

# LR---------------------------------------------------------------------
lr_model = LogisticRegression(random_state=0, max_iter=500)
lr_model.fit(X_train, y_train)

pred_lr = lr_model.predict(X_test) #X_test 내용으로 모델 만들기

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

print("\n--- Random Forest ---")
print(accuracy_score(y_test,pred))
print(confusion_matrix(y_test, pred))