from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_openml
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


# 미국 고등학생 및 인구조사국 직원들이 쓴 70,000개의
# 작은 숫자 이미지를 모은 MNIST 데이터셋
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape
# 70000개의 이미지, 각 이미지는 784개의 특성을 지니고 있다
# 왜냐하면 하나의 이미지가 28*28 px 이기 때문이다
# 개개의 특성은, 0(흰색) 부터 255(검은색) 까지의 픽셀 강도를 나타낸다


%matplotlib inline

# 샘플의 특성 벡터 추출
some_digit = X[0]
# 28 * 28 배열로 바꾸기
some_digit_image = some_digit.reshape(28, 28)
# 해당 이미지 그리기
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

save_fig("some_digit_plot")
plt.show()

y[0]  # 실제 5로 레이블 된 것을 확인할 수 있다

# 앞쪽 60000개는 훈련용 세트
# 뒤쪽 10000개는 테스트 세트
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


# 이진 분류기 ---
# 5이냐 5가 아니냐
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# Stochastic Gradient Descent 경사하강법 분류기
# 훈련시 "무작위성"을 사용한다
# 결과를 재현하고 싶다면 random_state 매개변수를 지정해야 한다

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([X[0]])
# 해당 분류기는 X[0]을 5라고 예측했다

# 성능 측정 ---
# 1) 정확도
# Fold가 3개인 K0겹 교차검증 사용하기
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
# 정확도가 95% 이상인 것을 알 수 있다
# 정확히 예측한 비율
# 하지만, 불균형한 데이터셋에서는, 정확도가 분류기의 성능 측정 지표로 좋지 않다

# 2) 오차행렬
# Confusion Matrix
# 오차행렬 활용하기
# 쉽게 말해 클래스 A 샘플이, B로 잘못분류된 횟수
# 숫자 5 이미지가 3으로 잘못 분류한 횟수를 알고 싶다면 --> 5행 3열

# 오차행렬을 만드려면, 실제 타깃과 비교할 수 있도록
# 먼저 예측값을 만들어야 한다 cross_val_predict()를 활용한다

# corss_val_predict의 경우 ,k-교차 검증을 수행하지만
# 평가점수를 반환하지 않고, 각 테스트 폴드에서 얻은 예측을 반환한다
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

confusion_matrix(y_train_5, y_train_pred)
# 오차행렬의 행은, 실제 클래스를 나타내고
# 열은 예측한 클래스를 나타낸다
# 5가 아닌 이미지를, 5가 아닌 것으로 정확히 분류한 것이 53892 개이다.
# 5가 아닌 이미지를, 5라고 분류한 것은 687
# 5인 이미지를, 5가 아니라고 분류한 것이 1891
# 5인 이미지를 5라고 분류한 것이 3530


# 3) 정밀도
# 5라고 예측된 애들 중에서 실제 5인 애들의 비율
# 3530 / 687 + 3530

# 4) 재현율 : recall
# 실제 5인 애들 중에서, 5차고 예측된 애들의 비율
# 3530 / 1891 + 3530

print("정밀도 :", precision_score(y_train_5, y_train_pred))
print("재현율 :", recall_score(y_train_5, y_train_pred))

# F1 점수 : 정밀도, 재현율 사이의 조화평균을 활용한다
# 이것이 항상 바람직한 것은 아니다
# 경우에 따라 , 정밀도가 중요할 수도 있고, 재현율이 중요할 수도 있다
f1_score(y_train_5, y_train_pred)

# 정밀도와 재현율은 서로 trade off 관계에 있다
# SGDClassifier 의 경우, 결정함수 Decision Function을 사용하여 각 샘플의 점수를 계산한다
# 이 점수가 임계점보다 크면, 샘플을 양성 클래스 True ( 5 )라고 할당하고
# 그렇지 않ㅇ느면, 음성 클래스 False( 5가 아님 )라고 할당한다

# 그렇다면 적절한 임계점(threshold)을 어떻게 정할 것인가
# 이를 위해서는, cross_val_predict() 함수를 사용하여, 훈련 세트에 있는
# 모든 샘플의 점수를 구해야 한다
# 하지만, 이번에는 위에서 했던 것처럼, 예측결과가 아니라
# 결정 점수를 반환받도록 지정할 것이다
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


# precision_recall_curve 함수를 사용하여
# 가능한 모든 임곗값에 대해 정밀도와 재현율을 계산할 수 있다
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
      plt.plot(thresholds, precisions[:-1], "b--", label="precision")
  plt.plot(thresholds,recalls[:-1],"g--",label="recall")
  plt.xlabel("threshold")
  plt.legend(loc="center left")
  plt.ylim([0,1])
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
# 정밀도 그래프가 울퉁불퉁한 이유는 가끔
# 임곗값을 올려도, 정밀도가 가끔 낮아질때가 있기 때문이다 

# 재현율 80% 근처에서 정밀도가 급격하게 줄어든다
# 이 하강점 직전을 정밀도/재현율 트레이드오프로 선택하는 것이 좋다

# 정밀도 90% 를 달성하는 것이 목표라고 가정해보자
# 임곗값이 약 7000 정도라는 것을 알 수 있다 


y_train_pred_90 = (y_scores > 7000)
print(precision_score(y_train_5,y_train_pred_90))
print(recall_score(y_train_5,y_train_pred_90))
# 정밀도가 83%인 분류기를 만들엇다
# 하지만, 참고로 재현율이 너무 낮으면, 유용하지 않게 된다  

# ROC 곡선
# ROC 곡선은, 정밀도에 대한 재현율 곡선이 아니라, 거짓 양성비율(FPR) 에 대한 진짜 양성 비율(TPR = 재현율 )이다
# FPR :실제 양성인 애들 중에서, 음성으로 잘못 분류된 애들의 비율 
# FPR = 1 - TNR ( 진짜 음성 비율 == 진짜 음성인 애들 중에서 , 음성으로 분류된 애들의 비율 == 특이도 )
# 따라서, ROC 곡선은, 민감도(재현율)에 대한 1-특이도 . 그래프이다
from sklearn.metrics import roc_curve
fpr,tpr,thresholds = roc_curve(y_train_5,y_scores)

def plot_roc_curve(fpr,tpr,label = None) :
  plt.plot(fpr,tpr,linewidth=2,label=label)
  plt.plot([0,1],[0,1],'k--')
  plt.axis([0,1,0,1])
  plt.xlabel('False Predicted Among True')
  plt.ylabel('True Predicted Among True')
plot_roc_curve(fpr,tpr)
plt.show()
# 좋은 분류기는 , 가운데 점선으로부터 최대한 멀리 떨어져 있어야 한다 


# AUC : 곡선 아래의 면적
# 완벽한 뷴류기는 ROC의 AUC가 1이고, 완전한 랜덤 분류기는 0.5 이다
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores)

# RandomForestClassifire 훈련시키기
from sklearn.ensemble import RandomForestClassifier

# 훈련 세트의 샘플에 대한 점수를 얻어야 한다
# 하지만, 작동방식의 차이로, decision_function이 없다 . 대신 predict_proba 메서드가 있다
# 샘플이 행이고, 클래스가 열이고, 샘플이 주어진 클래스에 속할 확률을 담은 배열을 반환한다 ( ex. 어떤 이미지가 5일 확률 70% )
forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3,
                                   method = 'predict_proba')
y_probas_forest

# ROC 곡선을 그리려면, 확률이 아니라, 점수가 필요하다
# 간단한 해결방법은, 양성 클래스의 확률을 점수로 사용하는 것이다 
y_scores_forest = y_probas_forest[:,1]
fpr_forest,tpr_forest,thresholds_forest = roc_curve(y_train_5,y_scores_forest)


plt.plot(fpr,tpr,"b:",label="SGD")
plot_roc_curve(fpr_forest,tpr_forest,"Random Forest")
plt.legend(loc="lower right")
plt.show() # 훨씬 더 좋은 것을 알 수 있다

# 마찬가지로 AUC 점수도 훨씬 높다
roc_auc_score(y_train_5,y_scores_forest)

# - 이진 뷴류기를 훈련시키는 방법 
# - 작업에 맞는 적절한 지표 선택
# - 교차 검증을 사용한 평가
# - 요구사항에 맞는 정밀도/재현율 트레이드 오프 선택
# - ROC 곡선, ROC AUC 점수를 사용한 여러 모델 비교 
