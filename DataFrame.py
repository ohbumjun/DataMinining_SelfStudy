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
# df = pd.read_csv(friend_src,encoding='utf-8',engine='python')
df = pd.read_csv(friend_src, encoding='utf-8')

# head() 데이터를 읽어보기 : 상위 5개
df.head()  # df.head(6) -- 6개 ! ( 설정 가능 )

# 데이터 저장하기
new_friend_src = base_src + '/new_friend.csv'

# df를 csv로 보낸다. 어디에 ?
# new_freind_src 라는 위치로 new_freind.csv 파일에 저장
# index = False는 무조건 !( index 1,2,3,4 에 해당하는 row 가 생겨버림 )
df.to_csv(new_friend_src, index=False, encoding="utf-8")

# Series 실습하기

# np와 pd의 차이
# 1) np는 index가 없고, pd는 index가 있다
# 2) np는 연산에 최적화 되어 있다
# np.array([1,2,3])

# 조회
# pd.Series([1,2,3])
# pd.Series([1,2,3],index=['가','나','다'])

# 데이터 프레임(집합) --> 시리즈(단일 - 한줄) --> 데이터 프레임
series = df['name']
series

# pd.Series 옵션이 무엇이 있는지 확인하기
# 1) index --> 중복 가능 ex) ['a','a','b']
# 2) dftype --> 대표적 : int, float,, 등등
series = pd.Series([1, 2, 3, 4], index=['a', 'a', 'c', 'd'], dtype=float)
series

series = pd.Series([10, 2, 5, 4, ], index=['a', 'b', 'c', 'd'], dtype=float)
series = series.sort_values(ascending=True)

series.sort_values(ascending=False)


# --------------------------------
# DataFrame 실습
# 1) dictionary 형태 : column 순서로 쌓인다
# columns : 열의 순서 지정해주기
df = pd.DataFrame({'a': [2, 3], 'b': [5, 10]}, columns=['b', 'a'])

# 2) array  :  row 순서로 쌓인다
df = pd.DataFrame([[2, 5, 100], [3, 10, 200], [10, 20, 300]],
                  columns=['a', 'b', 'c'],
                  index=['가', '나', '다'],
                  dtype=float)
df

# DataFrame 행,열 선택 및 필터링
df = pd.read_csv(friend_src, encoding='utf-8')
df.head()

# index 2번에 해당하는 row 가져오기
# 한 col 혹은 row를 가져오면 seires 가 된다
df.iloc[4]

# loc와 iloc
# dataframe 형태로 가져오기 == select * from friend where pk = 2
df.iloc[[4]]

# column 만 가져오가
df['job']

# column을 조금더 정형화된 형태로 가져오기
df.loc[:, 'job']
# df.iloc[:,2]

# iloc와 loc의 차이
# 1)
# iloc : 숫자 형태로 key 인식
# loc  : 문자 형태로 인식

# 2) idx
# iloc : idx 0번 기준으로 세팅
# loc : dataframe에 지정된 idx 기준으로 세팅
# ex) [1,2,3,4,5]로 idx를 지정했다면, 실제 1,2,3,4,5 와 관련하여
# 해당 idx 정보를 가져온다
# 즉, df.loc[2,:] != df.iloc[2,:]

# 공통점 : 둘다 slicing 개념
# ':' = 전체 다 가져오는 개념
# ',' 을 기준으로 왼쪽은 row, 오른쪽은 column
# 1번 idx의 job을 가져오고 싶어
df.loc[1, 'job']
df.iloc[1, 2]

# 쉼표 없을 시, row 기준으로만 slicing 해온다
df.iloc[2:4]

# column만 할때는 왼쪽에 : 을 넣어야 한다
df.iloc[:, :2]

# 정리
# iloc는 인덱스와 컬럼을 리스트 배열로 선택하는 것
# loc는 인덱스와 컬럼을 문자로 선택하는 것
# 따라서, 상황에 맞게 선택하는 것이 중요하다

# ---------------------------------------------
# DataFrame 행, 열 삭제하기
# 머신러닝 -> 차원의 저주 ... 가지치기 ... 대체값 처리(ex. 빈곳 처리하기)
# 차원의 저주 ? 변수가 너무 많아지면, 쓸모없는 변수가 있을수도 있다
# -- 이러한 것들을 모두 학습하게 되면.. 과적합이 일어나거나 등등
# -- 따라서, 차원 축소를 통해, 유의미한 변수만 뽑아내는 방법을 사용하곤 한다

# reshape : 3 * 4 로 만들어주겠다
df = pd.DataFrame(np.arange(12).reshape(3, 4),
                  columns=['A', 'B', 'C', 'D'])
# B,C column 삭제하기
# axis
# 1 : 세로 방향
# 0 : 가로 방향  ( default )
# inplace : 변수 재할당 안하고 현재 전처리 내용 그대로 적용
df.drop(['B', 'C'], axis=1, inplace=True)

# 데이터 수정하기
df.loc[1, 'C'] = '육'
# df.iloc[1,2] = '육'

# 슬라이싱 기능을 통해 범위를 바꾸기
df.iloc[:, 1] = "끝~"
df
