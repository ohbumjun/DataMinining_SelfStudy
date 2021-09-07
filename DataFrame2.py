# 데이터 프레임 행,열 선택 및 필터링 복습
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
