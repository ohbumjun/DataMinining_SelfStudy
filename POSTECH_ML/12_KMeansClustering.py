# unsupervised learing
# x 데이터만 존재, partition 개수만 주어져야 한다

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# data generation
# - np.random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)
# - 다변량 정규 분포에서 랜덤 표본을 추출
# - np.eye(2) : 2차원 단위행렬
G0 = np.random.multivariate_normal([1, 1], np.eye(2), 100)
G1 = np.random.multivariate_normal([3, 5], np.eye(2), 100)
G2 = np.random.multivariate_normal([9, 9], np.eye(2), 100)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)
print(X.shape)

plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'b.')
plt.axis('equal')
plt.show()

# number of clustes and data
k = 3
m = X.shape[0]

# randomly initialize mean points
mu = X[np.random.randint(0, m, k), :]  # 기존 데이터 범위 내에서 3개 랜덤 픽
pre_mu = mu.copy()  # mu를 update해갈 것이므로 ,prev 버전 마련한다
print(mu)

plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'b.')
plt.plot(mu[:, 0], mu[:, 1], 's', color='r',)
plt.axis('equal')
plt.show()

# clustering을 위한 임의 label
y = np.empty([m, 1])

# Run k-means
# 500번 반복
for _ in range(500):
    for i in range(m):
        # 모든 데이터 포인트의 거리 계산
        d0 = np.linalg.norm(X[i, :] - mu[0, :], 2)  # 첫번째 중심과의 차
        d1 = np.linalg.norm(X[i, :] - mu[1, :], 2)  # 두번째 중심과의 차
        d2 = np.linalg.norm(X[i, :] - mu[2, :], 2)  # 세번째 중심과의 차
        y[i] = np.argmin([d0, d1, d2])  # idx 를 return 한다

    err = 0
    for i in range(k):
        # ex) i가 0이면, 0에 해당하는 idx 목록만 가져온다
        # 그리고 평균을 취한다.
        # 해당 결과로 mu를 update 한다
        mu[i, :] = np.mean(X[np.where(y == i)[0]], axis=0)
        err += np.linalg.norm(pre_mu[i, :] - mu[i, :], 2)

    # err가 특정 값 이하라면 break
    pre_mu = mu.copy()  # mu가 update 되지 않으면 stop
    if err < 1e-10:
        break


# 각각 cluster 들의 좌표들
X0 = X[np.where(y == 0)[0]]
X1 = X[np.where(y == 1)[0]]
X2 = X[np.where(y == 2)[0]]

plt.figure(figsize=(10, 8))
plt.plot(X0[:, 0], X0[:, 1], 'b.', label='C0')
plt.plot(X1[:, 0], X1[:, 1], 'g.', label='C1')
plt.plot(X2[:, 0], X2[:, 1], 'r.', label='C2')
plt.axis('equal')
plt.legend(fontsize=12)
plt.show()


# skkit=learn

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

plt.figure(figsize=(10, 8))
plt.plot(X[kmeans.labels_ == 0, 0],
         X[kmeans.labels_ == 0, 1], 'b.', label='C0')
plt.plot(X[kmeans.labels_ == 1, 0],
         X[kmeans.labels_ == 1, 1], 'g.', label='C1')
plt.plot(X[kmeans.labels_ == 2, 0],
         X[kmeans.labels_ == 2, 1], 'r.', label='C2')
plt.axis('equal')
plt.legend(fontsize=12)
plt.show()

# Issue : Number of Cluster 문제
# 여러개의 K를 시도하기

# data generation
G0 = np.random.multivariate_normal([1, 1], np.eye(2), 100)
G1 = np.random.multivariate_normal([3, 5], np.eye(2), 100)
G2 = np.random.multivariate_normal([9, 9], np.eye(2), 100)

X = np.vstack([G0, G1, G2])
X = np.asmatrix(X)

cost = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    cost.append(abs(kmeans.score(X)))

plt.figure(figsize=(10, 8))
plt.stem(range(1, 11), cost)
plt.xticks(np.arange(11))
plt.xlim([0.5, 10.5])
plt.grid(alpha=0.3)
plt.show()
