from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os

PROJECT_ROOT_DIR = './drive/MyDrive/machine_learning'
TITANIC_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", "titanics")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")
# 훈련데이터를 이용해서, 가능한 최고의 모델을 만들고
# 테스트 데이터에 대한 예측을 캐글에 업로드하여 최종 점수 확인

'''
Column 설명
Survived: 타깃입니다. 0은 생존하지 못한 것이고 1은 생존을 의미합니다.
Pclass: 승객 등급. 1, 2, 3등석.
Name, Sex, Age: 이름 그대로 의미입니다.
SibSp: 함께 탑승한 형제, 배우자의 수.
Parch: 함께 탑승한 자녀, 부모의 수.
Ticket: 티켓 아이디
Fare: 티켓 요금 (파운드)
Cabin: 객실 번호
Embarked: 승객이 탑승한 곳. C(Cherbourg), Q(Queenstown), S(Southampton)
'''

# 누락된 데이터 확인하기
train_data.info()

# Age,Cabin,Embared 속성 일부가 null이다
# 891개의 non-null 보다 작다
# 특히나 Cabin은 대부분이 null
# 1) 따라서, Cabin은 일단 무시
# 2) 나이의 경우, 19% 는 null, 중간나이로 바꾸기
# 3) Name, Ticket은, 고려하기 까다롭기 때문에, 우선 제외

# 통계치 살펴보기 ---
train_data.describe()  # 38% 만 survived


# 타깃이 0과 1로 이루어졌는지 확인 ( 죽음 or 생존 ) ---
train_data["Survived"].value_counts()


# 범주형 특성들 보기 ---
train_data["Pclass"].value_counts()
train_data["Sex"].value_counts()
train_data["Embarked"].value_counts()


# 전처리 파이프라인 만들어보기

# 아래의 함수를 통해, DataFrame에서
# 특정 열을 선택하기
# 각열을 다르게 전처리 하기 위함이다
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


# Numeric Value를 위한 파이프라인 만들기

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median"))
])

# 해당 데이터 변환하기
num_pipeline.fit_transform(train_data)

# 문자열로된 범주형 열을 위해서
# 해당 열을 별도로 처리하는 Imputer 가 필요하다
# 일반 SimpleImputer 클래스는 이를 처리할 수 없다


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
      # c : column에 해당
      # X[c] : X data 의 c column
      # X[c].value_counts() : 해당 column 에서, 각 row에 대해 개수를 기준으로 내림차순
      # X[c].value_counts().index[0] : 가장 max값 가져오기
      # index = X.columns : 기존 X의 index column을 row로 사용하겠다
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# OneHotEncoder의 경우, 범주형 변수들을, 0,1 만을 가진 dummy variable로 변환하는 것이다

cat_pipeline = Pipeline([
    # 범주형 변수 열들을 선택한다
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    # 가장 많은 수를 가진 녀석으로 변환
    ("imputer", MostFrequentImputer()),
    # 이상변수화
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

# 실제 범주형 변수들도, 데이터분석이 가능한, 수치형 변수들로 변환하기
cat_pipeline.fit_transform(train_data)

# 숫자와 범주형을 연결한다
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = preprocess_pipeline.fit_transform(train_data)
X_train

# 예측할 label을 가져온다
y_train = train_data["Survived"]

# 분류기 훈련시키기

svm_clf = SVC(gamma="auto")


# 이를 이용해서 예측하기
X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)


# 교차 검증을 통해, 모델이 얼마나 좋은지 평가하기

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
# 약 73% --> 80% 까지 올리기

# RandomForestClassifier를 활용하기

forest_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
