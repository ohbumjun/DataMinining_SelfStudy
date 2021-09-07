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
