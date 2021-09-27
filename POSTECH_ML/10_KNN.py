# KNN Regression
# 가장 가까운 K개를 뽑아서 결론을 내리는 과정

from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

N = 100
w1 = 0.5
w0 = 2
x = np.random.normal(0, 15, N).reshape(-1, 1)
y = w1*x + w0 + 5*np.random.normal(0, 1, N).reshape(-1, 1)

plt.figure(figsize=(10, 8))
plt.title('Data set', fontsize=15)
plt.plot(x, y, '.', label='Data')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.axis('equal')
plt.axis([-40, 40, -30, 30])
plt.grid(alpha=0.3)
plt.show()

reg = neighbors.KNeighborsRegressor(n_neighbors=1)
reg.fit(x, y)

x_new = np.array([[5]])
pred = reg.predict(x_new)[0, 0]
print(pred)

# knn Regression
xp = np.linspace(-30, 30, 100).reshape(-1, 1)
yp = reg.predict(xp)

plt.figure(figsize=(10, 8))
plt.title('k-Neares Neighbor Regression', fontsize=15)
plt.plot(x, y, '.', label='Original Data')
plt.plot(xp, yp, label='kNN')
plt.plot(x_new, pred, 'o', label='Prediction')
plt.plot([x_new[0, 0], x_new[0, 0]], [-30, pred], 'k--', alpha=0.5)
plt.plot([-40, x_new[0, 0]], [pred, pred], 'k--', alpha=0.5)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.axis('equal')
plt.axis([-40, 40, -30, 30])
plt.grid(alpha=0.3)
plt.show()

# knn Classification
m = 1000

# 균등분포로부터 무작위 표본 추출
# np.random.uniform(low,high,size=[m,n,k])
# low 부터 hight 사이의 size 개수만큼 표본 추출
# 각각의 default value는 low가 0, high 가 1
# size : (m,n,k) --> m*n*k 개 만큼의 샘플 개수 추출
# 음, 여기서는 X가 m개의 행을 지는 2개짜리 열 (x,y ) 값을 의미하는 것 같다
X = -1.5 + 3*np.random.uniform(size=(m, 2))
# print(X[1,:])

# class 분류
y = np.zeros([m, 1])
for i in range(m):
    # X[i,:] : i번째 행 1열,2열
    # np.linalg.norm(X[i,:],2) : 해당 점과 (0,0) 사이의 거리
    if np.linalg.norm(X[i, :], 2) <= 1:
        y[i] = 1

# 즉, 원점으로부터 거리가 1 이내인 데이터들은 C1
# 그외는 C0으로 구분하겠다는 의미이다.

C1 = np.where(y == 1)[0]
C0 = np.where(y == 0)[0]

# 동그란 공간
theta = np.linspace(0, 2*np.pi, 100)

plt.figure(figsize=(8, 8))
plt.plot(X[C1, 0], X[C1, 1], 'o', label='C1',
         markerfacecolor='k',
         markeredgecolor='k',
         markersize=4)
plt.plot(X[C0, 0], X[C0, 1], 'o', label='C0',
         markerfacecolor="None",
         markeredgecolor='k',
         markersize=4)
plt.plot(np.cos(theta), np.sin(theta), '--', color='orange')
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()


# 1 : 가장 가까운 데이터의 class를 따라가기
clf = neighbors.KNeighborsClassifier(n_neighbors=1)
clf.fit(X, np.ravel(y))

# (0,0)을 predict 하려 한다
X_new = np.array([[0, 0]])
result = clf.predict(X_new)[0]
print(result)

res = 0.01
[X1gr, X2gr] = np.meshgrid(np.arange(-1.5, 1.5, res),
                           np.arange(-1.5, 1.5, res))

Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

inC1 = clf.predict(Xp).reshape(-1, 1)
inCircle = np.where(inC1 == 1)[0]

plt.figure(figsize=(8, 8))
plt.plot(X[C1, 0], X[C1, 1], 'o', label='C1',
         markerfacecolor='k',
         alpha=0.5,
         markeredgecolor='k',
         markersize=4)
plt.plot(X[C0, 0], X[C0, 1], 'o', label='C0',
         markerfacecolor='None',
         alpha=0.3,
         markeredgecolor='k',
         markersize=4)
plt.plot(Xp[inCircle][:, 0], Xp[inCircle][:, 1], 's',
         alpha=0.5,
         color='r',
         markersize=1)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.axis('equal')
plt.axis('off')
plt.show()
