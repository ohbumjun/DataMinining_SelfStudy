from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal


# kernel 매개변수가 "linear"일 때는 gamma가 무시됩니다.
param_distribs = {
    'kernel': ['linear', 'rbf'],
    'C': reciprocal(20, 200000),
    'gamma': expon(scale=1.0),
}

svm_reg = SVR()
# RandomizedSearchCv의 경우
# GridSearchCV와 거의 같은 방식으로 사용하지만
# 가능한 모든 조합을 시도하는 대신, 각 반복마다
# 하이퍼파라미터에 임의의 수를 대입하여 지정한 횟수만큼 평가한다
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# 5폴드 교차 검증 평가 점수
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse  # 54767.96071008413

# 최적의 하이어파라미터 확인하기
# {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}
rnd_search.best_params_
