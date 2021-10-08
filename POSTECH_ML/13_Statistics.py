import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# random parameter generation
m = 10

# unifrom distribution (0,1)
x1 = np.random.rand(m)  # m개 추출
print(x1.shape)

# uniform distribution (a,b)
a = 1
b = 5
x2 = a + (b-a) * np.random.rand(m)
print(x2)  # a 보다는 크고, b 보다는 작고


# normaldistrubition ( 0,1,m )
np.random.randn(m)

# normal distribution N (5,2^2)
# 평균 5 , 표준편차 2
5 + 2*np.random.randn(m)

# random int
np.random.randint(m)  # m 보다 작은 숫자 중에 random

# 1에서 6 사이의 10*1 행렬 ( 벡터 )
np.random.randint(1, 6, size=(m, 1))

# statistics
m = 100
x = np.random.rand(m, 1)

# 표본 평균
xbar = 1/m*np.sum(x, axis=0)
varbar = 1/(m)*np.sum((x-xbar)**2)  # 원래는 m-1개로 나눈다

print(xbar)
print(np.mean(x))
print(varbar)
print(np.var(x))

# various sample size m
m = np.arange(10, 2000, 20)  # 10부터 2000까지 20씩 증가
means = []
for i in m:
    # 10에서 30범위 사이의 정규분포에서 i개를 뽑는다
    x = np.random.normal(10, 30, i)
    means.append(np.mean(x))

# 뽑는 데이터의 개수가 많아질 수록
# 점차 원래의 평균에 가까워지는 것을
# 확인할 수 있다
plt.figure(figsize=(10, 6))
plt.plot(m, means, 'bo', markersize=4)
plt.axhline(10, c='k', linestyle='dashed')
plt.xlabel('# of samples ( = sample size)', fontsize=15)
plt.ylabel('sample mean', fontsize=15)
plt.ylim([0, 20])
plt.show()

# Central Limit Theorem
# 뽑는 데이터의 개수를 늘리면, 분산은 줄어든다
N = 100
m = np.array([10, 40, 160])   # sample of size m

S1 = []   # sample mean (or sample average)
S2 = []
S3 = []

for i in range(N):
    # uniform distribution 상에서, 10개의 데이터를 generate 한 것의 평균을 구한다
    # 즉, 표본 평균을 여러번 구하는 과정인 것이다
    S1.append(np.mean(np.random.rand(m[0], 1)))
    S2.append(np.mean(np.random.rand(m[1], 1)))
    S3.append(np.mean(np.random.rand(m[2], 1)))

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1), plt.hist(S1, 21), plt.xlim(
    [0, 1]), plt.title('m = ' + str(m[0])), plt.yticks([])
plt.subplot(1, 3, 2), plt.hist(S2, 21), plt.xlim(
    [0, 1]), plt.title('m = ' + str(m[1])), plt.yticks([])
plt.subplot(1, 3, 3), plt.hist(S3, 21), plt.xlim(
    [0, 1]), plt.title('m = ' + str(m[2])), plt.yticks([])
plt.show()

# Multivariate Statistic
# correlation coefficient

m = 300
x = np.random.rand(m)
y = np.random.rand(m)

xo = np.sort(x)
yo = np.sort(y)
yor = -np.sort(-y)

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'ko', label='random')
plt.plot(xo, yo, 'ro', label='sorted')
plt.plot(xo, yor, 'bo', label='reversely ordered')

plt.xticks([])
plt.yticks([])
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.axis('equal')
plt.legend(fontsize=12)
plt.show()

print(np.corrcoef(x, y), '\n')
print(np.corrcoef(xo, yo), '\n')
print(np.corrcoef(xo, yor))

# correlation coefficient

m = 300
x = 2*np.random.randn(m)  # x,y를 300개 uniform 하게 data 개수를 가져온다
y = np.random.randn(m)

xo = np.sort(x)
yo = np.sort(y)
yor = -np.sort(-y)  # reverset order

plt.figure(figsize=(8, 8))
plt.plot(x, y, 'ko', label='random')
plt.plot(xo, yo, 'ro', label='sorted')
plt.plot(xo, yor, 'bo', label='reversely ordered')

plt.xticks([])
plt.yticks([])
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.axis('equal')
plt.legend(fontsize=12)
plt.show()

print(np.corrcoef(x, y), '\n')   # x,y 간의 상관관계
print(np.corrcoef(xo, yo), '\n')  # 거의 1 인 것을 확인할 수 있다. 둘다 오름차순 정렬이므로
print(np.corrcoef(xo, yor))      # 반대 정렬

# 위 그래프와 ,아래 그래프의 유일한 차이점은 x 이다
# 즉, x의 개수에 있어서 차이가 난다는 것이다
# 그런데도 상관관계 계수는 거의 비슷한 값
# 즉, correlation 이란, 기울기는 거의 반영하지 않고
# 상관관계 그 자체만을 나타낸다는 것이다


d = {'col. 1': x, 'col. 2': xo, 'col. 3': yo, 'col. 4': yor}
df = pd.DataFrame(data=d)

sns.pairplot(df)
plt.show()
