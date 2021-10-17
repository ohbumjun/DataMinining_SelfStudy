# 축은
# 1) 선택하는 축의 분산은 최대
# 2) 버리는 축의 error는 최소

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# data generation
m = 5000
mu = np.array([0, 0])
sigma = np.array([[3, 1.5], [1.5, 1]])

# 다변량 정규 분포에서 랜덤 표본 추출
# sigma : covariance matrix
X = np.random.multivariate_normal(mu, sigma, m)
X = np.asmatrix(X)

flg = plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'k.', alpha=0.3)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# PCA
# Sample Covariance Matrix를 구해야 한다
S = 1/(m-1)*(X.T*X)  # 2*2
D, U = np.linalg.eig(S)  # 정사각형 배열의 고유값, 고유벡터 계산

idx = np.argsort(-D)  # 내림 차순
D = D[idx]
U = U[:, idx]

print(0, '\n')
print(D)  # eig Value
print(U)  # eig Vector


h = U[1, 0]/U[0, 0]  # 기울기
xp = np.arange(-6, 6, 0.1)  # -6부터 6사이의 0.1 간격 데이터 생성
yp = h * xp

flg = plt.figure(figsize=(10, 8))
plt.plot(X[:, 0], X[:, 1], 'k.', alpha=0.3)
plt.plot(xp, yp, 'r', linewidth=3)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()


Z = X * U[:, 0]  # U 자체가 transopose 된 형태로 구해졌으므로, 별도 transopose X
plt.figure(figsize=(10, 8))
plt.hist(Z, 51)
plt.show()

# 아래의 그림은, 위의 빨간색 선으로 projection하고
# 히스토그램을 그린 것이다


pca = PCA(n_components=1)  # 1개 축으로 줄인다
pca.fit(X)
Z = pca.transform(X)

plt.figure(figsize=(10, 8))
plt.hist(Z, 51)
plt.show()
