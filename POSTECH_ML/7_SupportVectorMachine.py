# 임의로 data를 generate 하기
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
