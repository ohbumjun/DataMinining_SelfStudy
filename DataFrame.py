import numpy as np
import os
import pandas as pd
# '.' : 현재 어떤 파일들이 있는지를 확인하기
# './drive ' : drive 파일로
os.listdir('./drive/MyDrive/machine_learning')

# 데이터 폴더 src 변수 할당
base_src = './drive/MyDrive/machine_learning'
# friend.csv 파일 src 변수 할당
friend_src = base_src + '/friend.csv'

# pandas의 read_csv => 데이터 불러오기
df = pd.read_csv(friend_src, encoding='utf-8', engine='python')

# head() 데이터를 읽어보기 : 상위 5개
df.head()  # df.head(6) -- 6개 ! ( 설정 가능 )

# 데이터 저장하기
new_friend_src = base_src + '/new_friend.csv'

# df를 csv로 보낸다. 어디에 ?
# new_freind_src 라는 위치로 new_freind.csv 파일에 저장
# index = False는 무조건 !( index 1,2,3,4 에 해당하는 row 가 생겨버림 )
df.to_csv(new_friend_src, index=False, encoding="utf-8")

os.listdir(base_src)

# Series 실습하기
