from sklearn.model_selection import GridSearchCV

# kernel의 개념
# kernel은 쉽게 말해서, 일반적으로 non-linear regression 혹은 classification을 하는데에 있어서
# 차원을 높여서, linear regression, classification을 수행한 이후
# 다시 차원을 낮춰서, 원래 데이터에 대한 non-linear regression, classificaiton 을 찾는 과정이다

# rbf 함수란, 정규분포 형태를 가진 여러 함수들을 basis로 하여
# 해당 함수들의 combination을 통해, 데이터에 대한 target function을 찾는 과정이라고 할 수 있다
param_grid = [
    {'kernel': ['linear'], 'C': [10., 30., 100.,
                                 300., 1000., 3000., 10000., 30000.0]},
    {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
     'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
]

svm_reg = SVR()
# 5 폴드 교차검증으로 평가하기
# GridSearchCV 의 교차검증기능은, scoring 매개변수에
# 낮을 수록 좋은, 비용함수가 아니라, 클수록 좋은 효용함수를 활용한다
# 즉, 클수록 좋은 것이다
grid_search = GridSearchCV(svm_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           verbose=2)
grid_search.fit(housing_prepared, housing_labels)

# 최상 모델로 평가한 점수 결과
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse  # 결과 : 70363.8400667152

# 최상의 parameter 확인하기
grid_search.best_params_
# {'C': 30000.0, 'kernel': 'linear'}
