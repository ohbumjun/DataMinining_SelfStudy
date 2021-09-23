# https://www.youtube.com/watch?v=9yl6-HEY7_s

# Build a predictor function to predict price of home
# 1) with 3400 sqr fit area in west windsor
# 2) 2800 sqr ft home in robbinsville

# 문제 : column 중에서의 text data를 handle해야 한다
# Categorical Variable에는 2가지 종류가 있다
# 1) Nominal -- numeric ordering이 없는 것
# ex) green,red,blue
# - 서로 간의 numeric relationship이 없다

# 2) Ordinal - numeric ordering이 있는 것
# ex) graduate, masters, phd
# ex) high, medium, low

# One Hot Encoding ?
# 새로운 column을 만들고, 각각의 text column에 대해서
# binary data를 할당한다 ( 0,1 )

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Suppress Warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')


# read data set
base_src = './drive/MyDrive/machine_learning/datasets/codebasics'
friend_src = base_src + '/homeprices.csv'
df = pd.read_csv(friend_src)
df

# dummy 3개 column을 return 한다
# 왜 ? 기존 data에서 하나의 column이 catergorical 이었고
# 3개의 category들을 가지고 있었기 때문이다
dummies = pd.get_dummies(df.town)
dummies

# axis = 'columns' : 오른쪽으로 붙여서 합친다
merged = pd.concat([df, dummies], axis='columns')
merged

# 기존 text column + 하나의 dummy variable은 drop 해야 한다
# 어떤 것을 택해도 상관 x
final = merged.drop(['town', 'west windsor'], axis='columns')
final

# create linear regression model
model = LinearRegression()

# price는 종속변수이므로, data set 에서 제거한다
X = final.drop('price', axis='columns')
X
y = final.price
y

model.fit(X, y)

# price of 2800 sqr fit 'home' in 'robbinsville'
model.predict([[2800, 0, 1]])


# 3400 sqr fit area in west windsor ( 두개 dummy에다가 둘다 0을 줘야 함을 의미한다 )
model.predict([[3400, 0, 0]])

# how accurate model is
model.score(X, y)

# LabelEncoder -------------------------------------
le = LabelEncoder()

dfle = df
# town column 내 각 항목을 label한 data들을 print 해줄 것이다
dfle.town = le.fit_transform(dfle.town)
dfle

# Crate X ( training data set)
X = dfle[['town', 'area']].values
y = dfle.price

# create dummy variable

# categorical_features = [0] : 0번째 idx가 categorical feature라고 명시
# 이를 해주지 않는다면, 모든 column이 categorical feature 라고 예측하게 할 것
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder='passthrough'
)

# 기존 X dataset의 0번째 column을 , 3개의 dummy variable을 만들어서
# 새로운 형태의 dataset을 return 한다
X = ct.fit_transform(X)
X

# 3개의 dummy variable 중 하나를 drop한다 ( dummy variable trap 방지 위해)
X = X[:, 1:]  # 앞 : = 모든 row 선택 // , 1: = 1번째 idx column부터 선택
X

model.fit(X, y)

# monroe, windsor, robinsville 중에서
# 첫번째를 지움,
# 이후, 지운 애들의 첫번째 column에서 1인 애들이 robbinsville 이었다
# price of 2800 sqr fit 'home' in 'robbinsville'
model.predict([[1, 0, 2800]])

# with 3400 sqr fit area in west windsor
model.predict([[0, 1, 3400]])
