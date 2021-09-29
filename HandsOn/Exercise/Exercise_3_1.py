# 1번
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

# KNN 알고리즘으로 학습시킨다
knn_clf = KNeighborsClassifier()

# 5 폴드 교차검증으로 평가하기
# GridSearchCV 의 교차검증기능은, scoring 매개변수에
# 낮을 수록 좋은, 비용함수가 아니라, 클수록 좋은 효용함수를 활용한다
# 즉, 클수록 좋은 것이다
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)


# 최적의 하이퍼 파라미터를 찾는다
grid_search.best_params_


# 최상의 모델로 평가한 점수
grid_search.best_score_

# 정확도 구하기

y_pred = grid_search.predict(X_test)
accuracy_score(y_test, y_pred)
