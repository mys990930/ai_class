import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
df = pd.read_csv("./csvs/titanic.csv")
print(df)
print(df.isnull().sum()) #결측치 확인
#df.dropna() #결측치 제거
drop_target = ["Name", "Ticket", "Cabin"]
df.drop(columns=drop_target, inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)
df.dropna(inplace=True)

print(df)
print(df.isnull().sum())

print("\n--- data 산점도 생성 ---")
plt.scatter(df['Survived'], df['Age'])
plt.xlabel('survived')
plt.ylabel('age')
plt.title("Survival state per age")
plt.grid(True)
plt.show()

print("\n--- box plot 생성 ---")
sns.boxplot(x="Age", data=df)
plt.title("Age")
plt.show()