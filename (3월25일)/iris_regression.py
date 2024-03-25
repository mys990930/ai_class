import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

TF_ENABLE_ONEDNN_OPTS = 0
columns_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("./csvs/iris.data", header=None, names=columns_name)

X = df[["sepal_width", "petal_length", "petal_width"]]
X.head()
y = df['sepal_length']
y.value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

# LR---------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
ly_preds = model.predict(X_test)

print('평균제곱근오차', mean_squared_error(ly_preds, y_test))
