import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./csvs/boston.csv", index_col=0)
df_corr = df.corr()

#feature selection을 위한 히트맵
plt.figure(figsize=(10,10))
sns.set(font_scale=0.8)
sns.heatmap(df_corr, annot=True, cbar=False)
plt.show()

X = df[['MedInc', 'AveRooms']] #feature 2개 select
X.head()
y = df['PRICE']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

# LR---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print('feature 2개인 경우: 평균제곱근오차', mean_squared_error(ly_preds, y_test))


#feature 1개, 3개인 경우와 비교--------------------------------------------------
X = df[['MedInc']]
X.head()
y = df['PRICE']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print('feature 1개인 경우: 평균제곱근오차', mean_squared_error(ly_preds, y_test))

X = df[['MedInc', 'HouseAge', 'AveRooms']]
X.head()
y = df['PRICE']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print('feature 3개인 경우: 평균제곱근오차', mean_squared_error(ly_preds, y_test))

