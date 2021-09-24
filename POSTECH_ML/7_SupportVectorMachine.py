# 임의로 data를 generate 하기
from sklearn import svm
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# training data generatioon
x1 = 8 * np.random.rand(100, 1)
x2 = 7 * np.random.rand(100, 1) - 4

g = 0.8 * x1 + x2 - 3
g1 = g - 1
g0 = g + 1

C1 = np.where(g1 >= 0)[0]
C0 = np.where(g0 < 0)[0]

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
ypt = -0.8 * xp + 3

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C!')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xp, ypt, 'k', linewidth=3, label='True')
plt.title('Linearly and Strictly Seperable Classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

# 자 이제 위의 직선을 모른다고 가정하고
# 위의 직선을 estimate 하려고 한다
xp = np.linspace(-1, 9, 100).reshape(-1, 1)
ypt = -0.8 * xp + 3

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C!')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xp, ypt, 'k', linewidth=3, label='True')  # 원래 직선
plt.plot(xp, ypt-1, '--k')  # -1
plt.plot(xp, ypt+1, '--k')  # +1
plt.title('Linearly and Strictly Seperable Classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

# First Attempt ------------------------------
# minimize somthing
# subject to
# 1) w0 + X1w >= 1
# 2) w0 + X0w <= -1

# 즉, data 들에서 boundary 까지의 거리 정보는 활용하지 않고
# 부호에 대한 정보만을 활용한다
# 그 결과는, 그저 boundary만을 찾아 줄뿐
# optimal한 line을 찾아주지는 않는다

# CVXPY using simple classification

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

N = X1.shape[0]
M = X0.shape[0]

w = cvx.Variable([2, 1])
w0 = cvx.Variable([1, 1])

obj = cvx.Minimize(1)
const = [w0 + X1*w >= 0, w0 + X0*w <= -1]
prob = cvx.Problem(obj, const).solve()

w0 = w0.value
w = w.value
print(w0)
print(w)

# 원래 직선에 plot 하기
xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[0, 0]/w[1, 0]*xp - w0/w[1, 0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C!')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xp, ypt, 'k', linewidth=1, label='True')  # 원래 직선
plt.plot(xp, ypt-1, '--k')  # -1
plt.plot(xp, ypt+1, '--k')  # +1
plt.plot(xp, yp, 'g', linewidth=3, label='Attempt')
plt.title('Linearly and Strictly Seperable Classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

# 축약된 모양

N = C1.shape[0]
M = C0.shape[0]

X1 = np.hstack([np.ones([N, 1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([M, 1]), x1[C0], x2[C0]])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

w = cvx.Variable([3, 1])

obj = cvx.Minimize(1)
const = [X1*w >= 0, X0*w <= -1]
prob = cvx.Problem(obj, const).solve()

w = w.value

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C!')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xp, ypt, 'k', linewidth=1, label='True')  # 원래 직선
plt.plot(xp, ypt-1, '--k')  # -1
plt.plot(xp, ypt+1, '--k')  # +1
plt.plot(xp, yp, 'g', linewidth=3, label='Attempt')
plt.title('Linearly and Strictly Seperable Classes', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

# 역시나 같에 나오는 것을 확인할 수 있다

X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])

N = X1.shape[0]
M = X0.shape[0]

X1 = np.hstack([np.ones([N, 1]), x1[C1], x2[C1]])
X0 = np.hstack([np.ones([M, 1]), x1[C0], x2[C0]])

outlier1 = np.array([1, 4.5, 1])
outlier2 = np.array([1, 2, 2])

X0 = np.vstack([X0, outlier1, outlier2])

X1 = np.asmatrix(X1)
X0 = np.asmatrix(X0)

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='C1')
plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='C0')
plt.title('When Outlier Exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

w = cvx.Variable([3, 1])

# 기존의 optimalization solution을 활용하면
# 해가 나오지 않는다
obj = cvx.Minimize(1)
const = [X1*w >= 1, X0 * w <= -1]
prob = cvx.Problem(obj, const).solve(solver='ECOS')

print(w.value)

# Second Attempt
# 이제는 relex를 시켜주려고 한다
# slack variable u,v를 추가한다
N = X1.shape[0]
M = X0.shape[0]

w = cvx.Variable([3, 1])
u = cvx.Variable([N, 1])
v = cvx.Variable([M, 1])

obj = cvx.Minimize(np.ones((1, N))*u + np.ones((1, M))*v)
const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0]
prob = cvx.Problem(obj, const).solve(solver='ECOS')

w = w.value

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]


plt.figure(figsize=(10, 8))
plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='C!')
plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='C0')
# 원래 직선 ( 우리가 맨 처음에 outlier 없이 그린 직선)
plt.plot(xp, ypt, 'k', linewidth=1, label='True')
plt.plot(xp, yp, 'g', linewidth=3, label='Attempt 2')
plt.plot(xp, yp-1/w[2, 0], '--g')
plt.plot(xp, yp+1/w[2, 0], '--g')
plt.title('When Outliers Exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()

# 진한 초록색을 새로 그리고
# 1개의 mis classifed된 outlier를 찾았다

# Maximize Margin
g = 3  # gamma

# gamma을 높인다는 것의 의미는 무엇일까 ?
# cvx.norm(w,2) + g*(np.ones((1,N))*u + np.ones((1,M))*v) 에 있어서
# 뒷쪽을 더 강조하겠다는 의미다
# 즉, 나는 뒷쪽에서 틀리는 것을 허용하지 않겠다
# X0*w <= -(1-v) 에서 v 크기를 작게 하겠다
# 즉, 폭을 작게 하겠다

# gamma을 줄인다는 것의 의미는 무엇일까 ?
# cvx.norm(w,2) + g*(np.ones((1,N))*u + np.ones((1,M))*v) 에 있어서
# 앞의 쪽을 더 강조한다는 것이다
# 반대로 말하면, 폭을 키우겠다는 의미이다


w = cvx.Variable([3, 1])
u = cvx.Variable([N, 1])
v = cvx.Variable([M, 1])

obj = cvx.Minimize(cvx.norm(w, 2) + g*(np.ones((1, N))*u + np.ones((1, M))*v))
const = [X1*w >= 1-u, X0*w <= -(1-v), u >= 0, v >= 0]
prob = cvx.Problem(obj, const).solve(solver='ECOS')

w = w.

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='C!')
plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='C0')
# 원래 직선 ( 우리가 맨 처음에 outlier 없이 그린 직선)
plt.plot(xp, ypt, 'k', alpha=0.5, label='True')
plt.plot(xp, ypt-1, '--k', alpha=0.2)
plt.plot(xp, ypt+1, '--k', alpha=0.2)
plt.plot(xp, yp, 'g', linewidth=3, label='SVM')
plt.plot(xp, yp-1/w[2, 0], '--g')
plt.plot(xp, yp+1/w[2, 0], '--g')
plt.title('When Outliers Exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()
# 바로 위 그래프보다 훨씬 더 나은 것을 확인할 수 있다.


# 조금 더 Compact 버전
# Compact Version
X = np.vstack([X1, X0])
# C1에 속하는 것은 1, C0에 속하는 것은 -1
y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

m = N + M

w = cvx.Variable([3, 1])
d = cvx.Variable([m, 1])

obj = cvx.Minimize(cvx.norm(w, 2) + g*(np.ones((1, m))*d))
const = [cvx.multiply(y, X*w) >= 1 - d, d >= 0]
prob = cvx.Problem(obj, const).solve(solver='ECOS')

w = w.value

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 1], X1[:, 2], 'ro', alpha=0.4, label='C!')
plt.plot(X0[:, 1], X0[:, 2], 'bo', alpha=0.4, label='C0')
# 원래 직선 ( 우리가 맨 처음에 outlier 없이 그린 직선)
plt.plot(xp, ypt, 'k', alpha=0.5, label='True')
plt.plot(xp, ypt-1, '--k', alpha=0.2)
plt.plot(xp, ypt+1, '--k', alpha=0.2)
plt.plot(xp, yp, 'g', linewidth=3, label='SVM')
plt.plot(xp, yp-1/w[2, 0], '--g')
plt.plot(xp, yp+1/w[2, 0], '--g')
plt.title('When Outliers Exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()
# 같은 결과가 나오는 것을 확인할 수 있다

# Skit-Learn Version --------------
X1 = np.hstack([x1[C1], x2[C1]])
X0 = np.hstack([x1[C0], x2[C0]])
X = np.vstack([X1, X0])

N = X1.shape[0]
M = X0.shape[0]

y = np.vstack([np.ones([N, 1]), np.zeros([M, 1])])


# clf : classifier
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

clf.predict([[6, 2]])

w = np.zeros([3, 1])
w[1, 0] = clf.coef_[0, 0]
w[2, 0] = clf.coef_[0, 1]
w[0, 0] = clf.intercept_[0]
print(w)

xp = np.linspace(-1, 9, 100).reshape(-1, 1)
yp = -w[1, 0]/w[2, 0]*xp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], 'ro', alpha=0.4, label='C!')
plt.plot(X0[:, 0], X0[:, 1], 'bo', alpha=0.4, label='C0')
# 원래 직선 ( 우리가 맨 처음에 outlier 없이 그린 직선)
plt.plot(xp, ypt, 'k', alpha=0.5, label='True')
plt.plot(xp, ypt-1, '--k', alpha=0.2)
plt.plot(xp, ypt+1, '--k', alpha=0.2)
plt.plot(xp, yp, 'g', linewidth=3, label='SVM')
plt.plot(xp, yp-1/w[2, 0], '--g')
plt.plot(xp, yp+1/w[2, 0], '--g')
plt.title('When Outliers Exist', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([0, 8])
plt.ylim([-4, 3])
plt.show()
# 같은 결과가 나오는 것을 확인할 수 있다


# 비선형
# Non-Linear SVM
# kernel을 통해서 mapping
# mapping된 공간 상에서 linear classification을 한 다음
# 원래 공간상에서 non-linear classification을
# 얻게 되는 것이다
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

# Kernel 준비하기
N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

X = np.asmatrix(X)
y = np.asmatrix(y)

m = N + M
# kernel function --> 열 벡터 형태로 나열하는 것이다
Z = np.hstack([np.ones([m, 1]), np.square(X[:, 0]), np.sqrt(
    2)*np.multiply(X[:, 0], X[:, 1]), np.square(X[:, 1])])

g = 10

w = cvx.Variable([4, 1])
d = cvx.Variable([m, 1])

obj = cvx.Minimize(cvx.norm(w, 2) + g*np.ones([1, m])*d)
const = [cvx.multiply(y, Z*w) >= 1-d, d >= 0]
prob = cvx.Problem(obj, const).solve()

w = w.value
print(w)

# 자. 우리는 mapping을 통해서 4차원으로 mapping 시켰다
# 그래프상 x,y 축 모두를 -3에서 3까지 0.1 간격으로 meshgrid를 뽑아낸다
# 해당하는 모든 점들이 x가 될 것이다
# numpy의 meshgrid함수는 1차원 좌표배열(x1,x2...xn) 에서 n차원
# 직사각형 격자를 만드는 함수이다.
[X1gr, X2gr] = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

m = Xp.shape[0]
Zp = np.hstack([np.ones([m, 1]), np.square(Xp[:, 0]), np.sqrt(
    2)*np.multiply(Xp[:, 0], Xp[:, 1]), np.square(Xp[:, 1])])
# 4차원 공간 상에서의 hyper plane
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
