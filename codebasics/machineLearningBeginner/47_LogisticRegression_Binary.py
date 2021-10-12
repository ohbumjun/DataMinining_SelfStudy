import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from matplotlib import pyplot as plt
%matplotlib inline

# read data set
base_src = './drive/MyDrive/machine_learning/datasets/codebasics'
friend_src = base_src + '/insurance_data.csv'
df = pd.read_csv(friend_src)
df.head()


plt.scatter(df.age, df.bought_insurance, marker='+', color='red')


X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], df.bought_insurance, train_size=0.8)

model = LogisticRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
y_predicted  # 1은, insurance 구매, 0은 구매 x
# 데이터에서 봤듯이, 나이가 많을 수록 insurance 구매 정도 높았음

model.score(X_test, y_test)
# 얼마나 정확한지 accuracy를 알려준다
# 데이터 사이즈가 작기 때문에 100%

model.predict_proba(X_test)
# 왼쪽은 0, 오른쪽은 1 ( not buy, buy )

# 기울기 : model.coef_ indicates value of m in y=m*x + b equation
# 절편   : model.intercept_ indicates value of b in y=m*x + b equation
print(model.coef_)
print(model.intercept_)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def prediction_function(age):
    z = 0.042 * age - 1.53  # 0.04150133 ~ 0.042 and -1.52726963 ~ -1.53
    y = sigmoid(z)  # 0 ~ 1 사이의 값으로 scaling
    return y


age = 35
prediction_function(age)

age = 43
prediction_function(age)  # ill buty the insurance
