# 4. Multivariate Regression and Nonlinear Regression

# 4_1) Multivariate Linear Regression
import numpy as np
import matplotlib.pyplot as plt

# for 3d plot
from mpl_toolkits.mplot3d import Axes3D

# y = theata0 + theta1*x1 + theta2*x2 + noise
n = 200
x1 = np.random.randn(n, 1)
x2 = np.random.randn(n, 1)
noise = 0.5*np.random.randn(n, 1)

y = 2 + 1*x1 + 3*x2 + noise

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('General Data', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, marker='.', label='Data')
# angle view 바꾸기 : 0,0
ax.view_init(30, 0)  # 위쪽에서 보는 것
plt.legend(fontsize=15)
plt.show()

# 배열을 왼쪽에서, 오른쪽으로 붙이기
A = np.hstack([x1**0, x1, x2])
A = np.asmatrix(A)

theta = (A.T*A).I * (A.T*y)

X1, X2 = np.meshgrid(
    np.arange(np.min(x1), np.max(x1), 0.5),
    np.arange(np.min(x2), np.max(x2), 0.5)
)

YP = theta[0, 0] + theta[1, 0] * X1 + theta[2, 0]*X2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_title('Regressopm', fontsize=15)
ax.set_xlabel('$X_1$', fontsize=15)
ax.set_ylabel('$X_2$', fontsize=15)
ax.set_zlabel('Y', fontsize=15)
ax.scatter(x1, x2, y, marker='.', label='Data')
# 사각형 그리기
ax.plot_wireframe(X1, X2, YP, color='k', alpha=0.3, label='Regression Plane')
# angle view 바꾸기 : 0,0
ax.view_init(30, 30)  # 위쪽에서 보는 것
plt.legend(fontsize=15)
plt.show()


# 4_2) NonLinear Regression
# 1) Polynomial Feature
n = 100
# 범위 : -5 ~ 10 사이
x = -5 + 15*np.random.rand(n, 1)
# 범위 : 0 ~ 10 사이
noise = 10*np.random.randn(n, 1)

# 실제 데이터
y = 10 + 1*x + 2*x**2 + noise

plt.figure(figsize=(10, 8))
plt.title('True x and y', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'o', markersize=4, label='actual')
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()


A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)
theta = (A.T*A).I*A.T*y
print("theta:\n", theta)  # 본래 값 ; 10,1,2

xp = np.linspace(np.min(x), np.max(x))  # plot을 하기위한 x,p의 pos
yp = theta[0, 0] + theta[1, 0]*xp + theta[2, 0]*xp**2

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', markersize=4, label='actual')
plt.plot(xp, yp, 'r', linewidth=2, label='estimated')
plt.title('Non linear regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim([np.min(x), np.max(x)])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()

# 2. Linear Basis Function Models
d = 6
xp = np.arange(-1, 1, 0.01).reshape(-1, 1)
polybasis = np.hstack([xp**i for i in range(d)])

plt.figure(figsize=(10, 8))

for i in range(d):
    plt.plot(xp, polybasis[:, i], label='$x^{}$'.format(i))

plt.title('Polynomial Basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([-1, 1, -1.1, 1.1])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()

# Non Linear Regression with Polynomial Functions
xp = np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1)

d = 3
polybasis = np.hstack([xp**i for i in range(d)])
polybasis = np.asmatrix(polybasis)

A = np.hstack([x**i for i in range(d)])
A = np.asmatrix(A)
theta = (A.T*A).I*(A.T*y)

yp = polybasis * theta

# 결과는 nonlinear regression과 동일
# 그저, basis 들의 linear combination 형태로 본다는 점에서
# 관점만을 달리하는 것이다
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Polynomial')
plt.title('Polynomial Basis', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis([-1, 1, -1.1, 1.1])
plt.grid(alpha=0.3)
plt.legend(fontsize=15)
plt.show()
