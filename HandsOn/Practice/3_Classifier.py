# 파이썬 ≥3.5 필수
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape  # 70000 : data set 개수 / 784 : height * width
# y : label data

y.shape

%matplotlib inline

# 하나의 digit을 정한다
some_digit = X[0]
# height,width 조정
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

# save_fig("some_digit_plot")
plt.show()

# 0 ~ 9 까지의 숫자만 필요( 그 사이에서 변화 )하기 때문에
# 아래와 같이 datatype을 바꾼다
y = y.astype(np.uint8)

# 그림을 보여주기 위한 함수


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

# 숫자 그림을 위한 추가 함수 ( multiple figures )


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9, 9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
# 100개의 data set
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 이진 분류기
# 이진 분류기 : input data가 5인지 아닌지를 구분할 것이다
# 0 : 5 // 1 : not 5
# define loss function : 쎄타 ( 모델링 parmater ) --> 쎄타 : parameter to estimate
# loss minimze 해야 쎄타를 잘 구할 수 있다 ( 실제 - x를 통해 예측된 값)
# 경사하강법을 통해 찾아낼 수 있다

# 문제는 minimize loss 하는 과정에서 많은 메모리를 차지하게 될 수 있다
# 이때는 mini-batch gradient descent 혹은 stochastic gradient descent를 활용한다

# stochastic gradient descent는 무엇일까 ?
# 매우 많은 데이터가 있다면 메모리가 부족
# 이때, 하나의 데이터들을, 여러개의 batch로 나눈다(묶음)
# 그리고 각각의 묶음에 대해서 gradient를 적용하는 것이다
#
# y_train_5 = (y_train == '5') # 5면 1, 아니면 0 으로 표시
y_test_5 = (y_test == '5')


# SGDClassifier 적용하기
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# some_digit이 5인지 아닌지를 예측
# 실제 5이고 , 예측도 5라고 한다
sgd_clf.predict([some_digit])

# cross validation 적용하기
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# Measuring Accuracy using cross-validation

# shuffle=False가 기본값이기 때문에 random_state를 삭제하던지 shuffle=True로 지정하라는 경고가 발생합니다.
# 0.24버전부터는 에러가 발생할 예정이므로 향후 버전을 위해 shuffle=True을 지정합니다.
# 기존 분포를 유지시켜준다
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    # accuracy를 보여준다 ( 하지만 accuray의 경우 , unbalanced 된 data일 때에 대비해야한다)
    # https://www.youtube.com/watch?v=qWfzIYCvBqo
    # 즉, 정확도는 믿을만한 판단 근거가 되지 못한다는 것이다
    # ex) 현재 우리 데이터 중에서는 90% 가 무려 not 5에 속한다
    # 즉, 5의 비율이 10% 인 것이다
    # 만약 9 classifer가 모두 , "모두가 not 5"라고 분류해버릴 수 있다
    # 그렇게 되면 accuracy는 90% 이다.
    # 즉, 이상한 classifer인데도 정확도는 90% 가 된다
    # 즉, General accuracy criteria is not good when class is unbalanced
    print(n_correct / len(y_pred))

    # https://www.youtube.com/watch?v=qWfzIYCvBqo
# Precision : positive라고 분류된 애들 중에서, 실제 positive를 본다
# Recall    : 실제 positive인 애들을 본다. 그중에서 positive로 분류된 애들
# f1 : Precision, Recall 은 서로 반비례 관계 --> 중첩점

# ---------
# ROC Curve :
# Confusion Matrix : https://radical-pony-f47.notion.site/5ee13fd9ea744e57a5307f64bbd96f0f


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 혼동행ㄹ려

confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5  # 완변한척 하자
confusion_matrix(y_train_5, y_train_perfect_predictions)  # 그렇게 되면, 대각원소만 존재


precision_score(y_train_5, y_train_pred)
# confusion matrix 상으로도 구할 수 있다
# precision : positive라고 분류된 애들 중에서 실제 positive
# 3530 / ( 687 + 3530  )

cm = confusion_matrix(y_train_5, y_train_pred)
cm[1, 1] / (cm[0, 1] + cm[1, 1])

recall_score(y_train_5, y_train_pred)


f1_score(y_train_5, y_train_pred)

# Decision function 활용하기
# sgd_clf.predict 를 쓰면, true or false 의 prediction이 나온다
# 반면, sgd_clf.decision_function을 쓰면 실제 결과값(real value)이 나온다
y_scores = sgd_clf.decision_function([some_digit])
y_scores

# threshold를 통해, 우리의 예측이 true인지 false인지도 구할 수 있다
threshold = 0
y_some_digit_pred = (y_scores > threshold)

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")


precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# precision, recall curve 그리기


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--",
             label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)  # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown


recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


# Not shown
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision],
         [0., 0.9], "r:")                 # Not shown
plt.plot([-50000, threshold_90_precision], [0.9, 0.9],
         "r:")                                # Not shown
plt.plot([-50000, threshold_90_precision],
         [recall_90_precision, recall_90_precision], "r:")  # Not shown
# Not shown
plt.plot([threshold_90_precision], [0.9], "ro")
plt.plot([threshold_90_precision], [recall_90_precision],
         "ro")                             # Not shown
# Not shown
save_fig("precision_recall_vs_threshold_plot")
plt.show()


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
save_fig("precision_vs_recall_plot")
plt.show()
# 파이썬 ≥3.5 필수
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "Linear Models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Linear Regression

# 모델을 훈련시킨다는 것은, 모델이 훈련 세트에 가장 잘 맞도록 모델
# 파라미터를 설정하는 것이다

# 이를 위해 먼저, 모델이 훈련 데이터에 얼마나 잘 들어맞는지 측정해야 한다
# RMSE를 통해, 회귀 성능 측정을 할 수 있다
# 즉, RMSE를 최소화 하는 파라미터들을 찾으면 된다

# 정규방정식.
# 즉, 해당 파라미터를 찾는 공식이 따로 존재한다 (노트북에 붙여둔 것 )


**식 4-4: 정규 방정식**

$\hat{\boldsymbol{\theta}} = (\mathbf{X} ^ T \mathbf{X}) ^ {-1} \mathbf{X} ^ T \mathbf{y}$

# 랜덤 데이터 생성하기

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
