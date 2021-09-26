# Logistic Regression은 전체 데이터를 활용한다

# 최대한 linear boundary를 center에 위치시키고 싶다
# 어떻게 ? 모든 점에서 boundary에 내린 h라는 거기를 구하고
# 산술기하 평균하에서, 그것들의 곱이 최대가 되는 오메가를 찾고자 하는 것이다

# 그런데 거리를 있는 그대로 활용하게 되면
# 이상치 outlier의 영향을 많이 받기 때문에
# sigmoid function을 활용하여, 그 범위를 제한해줄 것이다
# 0 ~ 1
# 데이터가 C1 에 속할 확률. 이라고도 정의할 수 있을 것이다

from sklearn import linear_model
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

z = np.linspace(-4, 4, 100)  # -4에서 4까지 100등분
s = 1/(1+np.exp(-z))

plt.figure(figsize=(10, 2))
plt.plot(z, s)
plt.xlim([-4, 4])
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# 0 근처에서는 선형적, 거리에 대한 것을 고려
# 0 에서 멀어지면, 거리를 일정하게 제한해서 고려한다
#
# # 우리의 최종적인 목표는 g(x) == 0을 찾는 것이다
# 즉, w(오메가)를 찾는 것이 우리의 목표이다
# m개의 데이터 x point 들마다
# 각각 x에 대응하는 y, 즉, label이 존재할 것이다.
# 각각의 x마다, 해당 y label을 가장 많이 맞추는 오메가 w를 구해야 한다
# 즉, 그렇다면, 각각의 점에 대한 확률을 , 모두 곱하고 ( 서로 독립 )
# 그 결과 확률값을 기준으로, decision making을 해야 한다.
# 그런데, 확률은 0과 1 사이의 값을 지닌다
# 그렇게 많은 데이터를 곱해주게 되면, 최종 확률값은 매우 낮아질 것이다
# ( 소수값을 곱해주기 때문에 )
# 따라서, 최종 확률값을 구하는 공식 == 각각희 확률 곱하기.
# 를 활용하는 대신에, log를 취하여
# 각각의 확률값을 더하는 방식을 활용할 것이다

# data 만들기
m = 100
# 처음 임의의 w 값을 정한다
w = np.array([[-6], [2], [1]])
# [1,~,~] 형태의 0 ~ 4 사이의 값을 지니는 임의의 데이터를 만든다
X = np.hstack(
    [np.ones([m, 1]), 4*np.random.rand(m, 1), 4*np.random.rand(m, 1)])

w = np.asmatrix(w)
X = np.asmatrix(X)

# y가 0.5 이상인 애들은 1, 아닌 애들은 0
y = 1/(1 + np.exp(-X*w)) > 0.5

# y가 true인 애들의 idx
C1 = np.where(y == True)[0]
# y가 false인 애들의 idx
C0 = np.where(y == False)[0]

y = np.empty([m, 1])
y[C1] = 1
y[C0] = 0

plt.figure(figsize=(10, 8))
plt.plot(X[C1, 1], X[C1, 2], 'ro', alpha=0.3, label='C1')
plt.plot(X[C0, 1], X[C0, 2], 'bo', alpha=0.3, label='C0')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_w$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

# cvxpy를 이용
# reordering한다
# C1에 속하는 애들을 모으고, 그 이후에 C0에 속하는 애들을 모은다


w = cvx.Variable([3, 1])
obj = cvx.Maximize(y.T*X*w - cvx.sum(cvx.logistic(X*w)))
prob = cvx.Problem(obj).solve()

w = w.value

xp = np.linspace(0, 4, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(X[C1, 1], X[C1, 2], 'ro', alpha=0.3, label='C1')
plt.plot(X[C0, 1], X[C0, 2], 'bo', alpha=0.3, label='C0')
plt.plot(xp, yp, 'g', linewidth=4, label='Logistic Regression')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_w$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

# Compact Version
# 0과 1이 아니라
# -1과 1 사이의 범위로 세팅한다

y = np.empty([m, 1])
y[C1] = 1
y[C0] = -1
y = np.asmatrix(y)

w = cvx.Variable([3, 1])
obj = cvx.Maximize(cvx.sum(-cvx.logistic(cvx.multiply(-y, X*w))))
prob = cvx.Problem(obj).solve()

w = w.value

xp = np.linspace(0, 4, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(X[C1, 1], X[C1, 2], 'ro', alpha=0.3, label='C1')
plt.plot(X[C0, 1], X[C0, 2], 'bo', alpha=0.3, label='C-1')
plt.plot(xp, yp, 'g', linewidth=4, label='Logistic Regression')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

# Scikit-Learn 활용하기
X = X[:, 1:3]  # 원래 X에서 임의의 첫번째 요소를 제거한다
X.shape

clf = linear_model.LogisticRegression(solver='lbfgs')
clf.fit(X, np.ravel(y))

clf.predict([[1, 2]])  # -에 속한다
clf.coef_  # w1,w2
clf.intercept_  # w0

# 직선 계산 및 그래프 그리기
w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]

xp = np.linspace(0, 4, 100).reshape(-1, 1)
yp = -w1/w2*xp - w0/w2

plt.figure(figsize=(10, 8))
plt.plot(X[C1, 0], X[C1, 1], 'ro', alpha=0.3, label='C1')
plt.plot(X[C0, 0], X[C0, 1], 'bo', alpha=0.3, label='C-1')
plt.plot(xp, yp, 'g', linewidth=4, label='Logistic Regression')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 4])
plt.ylim([0, 4])
plt.show()

# Non-linear
# kerner을 활용한다
X1 = np.array([
    [-1.1, 0], [-0.3, 0.1], [-0.9, 1],
    [0.8, 0.4], [0.4, 0.9], [0.3, -0.6],
    [-0.5, 0.3], [-0.8, 0.6], [-0.5, -0.5]
])

X0 = np.array([[-1, -1.3], [-1.6, 2.2], [0.9, -0.7],
               [1.6, 0.5], [1.8, -1.1], [1.6, 1.6],
               [-1.6, -1.7], [-1.4, 1.8], [1.6, -0.9],
               [0, -1.6], [0.3, 1.7], [-1.6, 0], [-2.1, 0.2]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], 'ro', alpha=0.4, label='C!')
plt.plot(X0[:, 0], X0[:, 1], 'bo', alpha=0.4, label='C0')
plt.title('SVM for Nonlinear Data', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.show()

# 마찬가지로 kerner를 사용해서 mapping 하기
# Kernel 준비하기
N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

X = np.asmatrix(X)
y = np.asmatrix(y)

m = N + M
# kernel function --> 열 벡터 형태로 나열하는 것이다
Z = np.hstack([np.ones([m, 1]), np.sqrt(2)*X[:, 0], np.sqrt(2)*X[:, 1],
               np.square(X[:, 0]), np.sqrt(2)*np.multiply(X[:, 0], X[:, 1]),
               np.square(X[:, 1])])

w = cvx.Variable([6, 1])
d = cvx.Variable([m, 1])

obj = cvx.Minimize(cvx.sum(cvx.logistic(-cvx.multiply(y, Z*w))))
prob = cvx.Problem(obj).solve()

w = w.value
print(w)

# to plot
[X1gr, X2gr] = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

m = Xp.shape[0]
Zp = np.hstack([np.ones([m, 1]), np.sqrt(2)*Xp[:, 0], np.sqrt(2)*Xp[:, 1],
                np.square(Xp[:, 0]), np.sqrt(2) *
                np.multiply(Xp[:, 0], Xp[:, 1]),
                np.square(Xp[:, 1])])

# 6차원 공간 상에서의 hyper plane
q = Zp*w

# q가 0보다 큰 것을 B에 담아둘 것이다
# B안에 있는 것만 색상으로 grid 표시를 하면
B = []
for i in range(m):
    if q[i, 0] > 0:  # C1 에 해당하는 것
        B.append(Xp[i, :])

B = np.vstack(B)

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='C!')
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='C0')
plt.plot(B[:, 0], B[:, 1], 'gs', markersize=10, alpha=0.1, label='SVM')
plt.title('SVM for Kernel', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.show()
