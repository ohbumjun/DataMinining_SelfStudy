from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Suppress Warnings for clean notebook
import warnings
warnings.filterwarnings('ignore')

# read data set
base_src = './drive/MyDrive/machine_learning/datasets/codebasics'
friend_src = base_src + '/Melbourne_housing_FULL.csv'
dataset = pd.read_csv(friend_src)
dataset.head()

dataset.shape

# discard certain columns ex) Date,
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG',
               'Regionname', 'Propertycount', 'Distance',
               'CouncilArea', 'Bedroom2', 'Bathroom', 'Car',
               'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
dataset.head()

dataset.isna().sum()

# fill na with 0
cols_to_fill_zero = ['Propertycount',
                     'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)
dataset.isna().sum()

# 일부는 평균으로 채우기
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(
    dataset.BuildingArea.mean())

# 일부 na는 drop 하기
dataset.dropna(inplace=True)

# 모든 na 제거 성공
dataset.isna().sum()

# Test를 dummy data들을 columd으로 붙이고
# 새롭게 생긴 애 중에서 첫번째 column은 제거한다
dataset = pd.get_dummies(dataset, drop_first=True)
dataset.head()

X = dataset.drop('Price', axis=1)
y = dataset['Price']

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=2)

reg = LinearRegression().fit(train_X, train_y)

# test에 대해서는 성능이 낮고
reg.score(test_X, test_y)

# train에 대해서는 성능이 높다
# 즉, overfitting이 발생하고 있다는 것이다
reg.score(train_X, train_y)

# Lasso --> L1 Norm
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)

# test data set에 대해서 성능이 높아진 것을 확인할 수 있다
lasso_reg.score(test_X, test_y)

ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

# 마찬가지로 증가
ridge_reg.score(test_X, test_y)
