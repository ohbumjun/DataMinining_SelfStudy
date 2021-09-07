import math
import numpy as np
import os
import pandas as pd

base_src = './drive/MyDrive/machine_learning'
abalon_src = base_src + '/abalone.data'
abalon_df = pd.read_csv(abalon_src,
                        header=None,  # header를 주지마
                        sep=',',  # ','을 기준으로 나눌 거야
                        names=['sex', 'length', 'diameter',
                               'height', 'whole_weight',
                               'shucked_weight', 'viscera_weight',
                               'shell_weight', 'rings']  # column 이름을 names 라고 한다
                        )
abalon_df.head()

# 데이터 shape를 확인한다( data 형태 )
abalon_df.shape  # ( 4177, 9) : row가 4177 개, col가 9개


# 데이터 결측값 확인 ( 빈값 확인 )
# True : 1 , False : 0 --> isnull()이 True 라는 것은, 결측값을 의미
abalon_df.isnull().sum()  # 열 기준 확인
abalon_df.isnull().sum().sum()  # 전체 데이터 확인

# 기술 통계 확인하기 -> 연속형 변수만 확인할 수 있다 (numerical 한 변수)
abalon_df.describe()

# 왜 sex 에 대한 내용이 없을까 ?
# sex는 categorical 변수이기 때문이다( 남자, or 여자 )
# 그외 numerical 변수이다

# 집계함수 사용하기
# 전복(abalone) 성별에 따라 groupby 함수를 통해 집계하기
# DataFrame[집계변수].groupby(DataFrame[집계대상])
grouped = abalon_df['whole_weight'].groupby(abalon_df['sex'])
grouped.sum()

grouped.mean()  # 이상치의 영향을 받는다

grouped.size()  # 데이터 개수가 몇개인가

# 그룹변수가 하나가 아닌
# 전체 연속형 변수에 대한 집계
abalon_df.groupby(abalon_df['sex']).mean()

# 다음과 같이 간단히 표현가능
abalon_df.groupby('sex').mean()

# 새로운 조건에 맞는 변수(column) 추가하기

# abalon_df['length'] 가 median보다 큰 애들은 length_long, 작으면 length_short
abalon_df['length_bool'] = np.where(
    abalon_df['length'] > abalon_df['length'].median(),
    'length_long',
    'length_short')
abalon_df


# 그룹변수를 2개이상 선택해서 통계처리하기
# 1) 성별 대로 나누고 --> 2) 또 length로 나눈것
abalon_df.groupby(['sex', 'length_bool']).mean()

abalon_df.groupby(['sex', 'length_bool'])['whole_weight'].mean()

# -------------------------------------------------------
# 중복 데이터 확인하기
# 중복데이터 삭제 결측치 --> 중복된 데이터 확인하기

# 중복된 row 확인하기
abalon_df.duplicated()
abalon_df.duplicated().sum()  # 0 !

# 중복 예제 생성을 위해 --> 가상으로 중복데이터 생성
# pd.concat() : 데이터 붙이기
new_abalon = abalon_df.iloc[[0]]  # [[]] : data frame 형태로 가져오기

# abalon_df, new_abalone 을, 가로로 붙일 것이다 ( 아래 row에 붙인다 )
new_abalon_df = pd.concat([abalon_df, new_abalon], axis=0)

new_abalon_df.duplicated()  # 마지막 행 중복
new_abalon_df.duplicated().sum()  # 1

# 첫번째 중복 데이터 중에서, 처음애를 지우자
new_abalon_df.duplicated(keep='last')  # 마지막 애를 보존하자

# 중복 데이터 중에서, 첫번째 제외, 그 이후 중복애들 row 삭제
new_abalon_df = new_abalon_df.drop_duplicates()
# new_abalon_df = new_abalon_df.drop_duplicates(keep='last') : 마찬가지로 뒤에꺼 죽이기

# NaN(결측치)를 찾아서, 다른 값으로 변경하기
# 기존 데이터에는 결측치가 존재 x
abalon_df.isnull().sum()

# 가상으로 결측치 만들기
nan_abalon_df = abalon_df.copy()
# 값 변경
nan_abalon_df.loc[2, 'length'] = np.nan
nan_abalon_df.isnull().sum()

# 결측치를 특정 값으로 채우기

# 결측치를 0으로 채운다
zero_abalone_df = nan_abalon_df.fillna(0)
zero_abalone_df  # 실수로 바뀜 ( 우선순위 : 실수 > 정수 )

# 결측치를 결측치가 속한 컬럼의 평균값으로 대체하기
nan_abalon_df.mean()
nan_abalon_df = nan_abalon_df.fillna(nan_abalon_df.mean())

# 아래와 같이 특정 column 및 행렬에 대한 작업도 가능하다
nan_abalon_df['length'].fillna(nan_abalon_df['length'].mean())

# 카테고리 변수 결측치 처리
nan_abalon_df.loc[2, 'sex'] = np.nan
# 평균은 적용할 수 없으므로, 가장 많은 값으로 채우겠다
nan_abalon_df['sex'].value_counts().max()
# 혹은
sex_mode = nan_abalon_df['sex'].value_counts(
).sort_values(ascending=False).index[0]
nan_abalon_df['sex'].fillna(sex_mode)


# 꽃 ...!
# apply 함수 !!!! 진짜 강추
# 본인이 원하는 행과 열에 연산 혹은 function을 적용할 수 있다
# 열 기준으로 집계하고 싶은 경우 axis = 0
# 행 기준으로 집계하고 싶은 경우 axis = 1

# 1) 열 기준 집계
abalon_df[['diameter']].apply(np.average, axis=0)

# 2) 행 기준 집계
abalon_df[['diameter']].apply(np.average, axis=1)


# 사용자 정의함수를 통한 apply
def avg_ceil(x, y, z):
    return math.ceil((x+y+z)/3)  # ceil : 올림


abalon_df[['diameter', 'height', 'whole_weight']]
.apply(lambda x: avg_ceil(x[0], x[1], x[2]), axis=1)

# 문제


def is_sum_over_3(x, y, z):
    return (x+y+z) > 1


# ['diameter','height','whole_weight'] 변수 사용
# 세 변수의 합이 1이 넘으면 true, 아니면 false 출력 후 answer 변수에 저장
# abalon_df 에 answer 열을 추가하고 입력
abalon_df['answer'] = np.where(
    abalon_df[['diameter', 'height', 'whole_weight']]
    .apply(lambda x: is_sum_over_3(x[0], x[1], x[2]), axis=1),
    True,
    False)
abalon_df


# ---------------------------------------------------
# 카테고리 변수
# 컬럼내 유니크한 값 뽑아서 개수 확인하기 (카테고리 변수에 적용)
abalon_df['sex'].value_counts(ascending=True)
# abalon_df['sex'].value_counts(dropna = True) : 결측치 제외하고 보기


# --------------------------------------------------
# 2개의 data frame 합치기
# 가상 abalon 1개 row 데이터 생성 및 결합
one_abalon_df = abalon_df.iloc[[0]]
pd.concat([abalon_df, one_abalon_df], axis=0)

# 가상 abalon 1개 column 데이터 생성 및 결합
one_abalon_df = abalon_df.iloc[:, [0]]  # 성별 정보
pd.concat([abalon_df, one_abalon_df], axis=1)
