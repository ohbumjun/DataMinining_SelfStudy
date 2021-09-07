# 데이터 프레임 행,열 선택 및 필터링 복습
import numpy as np
import os
import pandas as pd

os.listdir('./drive/MyDrive/machine_learning')
base_src = './drive/MyDrive/machine_learning'
friend_src = base_src + '/friend.csv'
df = pd.read_csv(friend_src)
df.head()

# index 2번에 해당하는 해당 row 가져오기
df.iloc[2]
# column job에 해당하는 data 가져오기
df['job']
# column job에 해당하는 데이터 가져오기(2)
df.loc[:, 'job']
# 슬라이싱 기능을 통해 여러 행 가져오기
# df.iloc[[2,3]]
df.iloc[2:4]

# 조건 필터링 가져오기
# 30 대 이상의 사람들만 가져오기
df['age'] >= 30
df[df['age'] >= 30]

# job이 intern인 사람 가져오기
df['job'] == 'intern'
df[df['job'] == 'intern']

# 조건이 여러개 라면
# ex) 30대 이상 , 40대 이하 ( and function )
# df[df['age'] >= 30 & df['age'] <= 40]
# 위의 내용은 에러
# data frame 쓸 때는, 각각의 조건들에 대해서 ()을 한번 더 씌운다
# 이를 통해 value 형태로 만들어주는 것
df[(df['age'] >= 30) & (df['age'] <= 40)]

# ex) 30대 미만 , 혹은  40대 초과 ( or function )
df[(df['age'] < 30) | (df['age'] > 40)]

# in을 통한 포함 조건 걸기
# df[df['job'] in ['student','manager']] --> 에러
# 왜 ? df['job']은 하나의 값이 아니라, series ( 여러개)
# 따라서, 왼쪽에는 데이터 하나만 와야 한다
df[(df['job'] == 'student') | (df['job'] == 'manager')]

# df['job']에 apply 할거야
# x는 df['job'] 원소 하나 하나
# 그 x 중에서 'student' 혹은 'manager'인 애를 가져올 거야
df[df['job'].apply(lambda x: x in ['student', 'manager'])]

# DataFram 행,열 삭제
df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])
df

# B, C 컬럼을 삭제하고 싶다
df.drop(['B', 'C'], axis=1, inplace=True)
df

# index 0과 1을 삭제하겠습니다
df.drop([0, 1], axis=0)

# DataFrame 행과 열 수정하기
df = pd.read_csv(friend_src, encoding='utf-8')
# df를 하나 복사하기 ( 원본 데이터 유지를 위해서 )
temp = df.copy()

# age 컬럼 모든 값 변경
temp['age'] = 20  # 모든 열 데이터 전체 바뀌기
temp
# 지정한 idx의 column 바꾸기
temp.loc[2, 'age'] = 27
temp
