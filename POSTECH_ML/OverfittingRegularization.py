import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# 10 data points
n = 10
x = np.linspace(-4.5, 4.5, 10).reshape(-1, 1)
y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180,
              -0.8361, -0.6591, -2.4701, -2.8122, -6.2512]).reshape(-1, 1)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# Linear Regression 찾아내기
A = np.hstack([x**0, x])  # A matrix 만들어내기
A = np.asmatrix(A)

# optimal한 theta를 찾고
theta = (A.T*A).I * (A.T*y)
print(theta)


# 화면에 직선 그려내기
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0, 0] + theta[1, 0] * xp

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, linewidth=1, label='Linear')
plt.title('Linear Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 차원 확장시키기
A = np.hstack([x**0, x, x**2])
A = np.asmatrix(A)

theta = (A.T*A).I*(A.T*y)
print(theta)

# to plot
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)
yp = theta[0, 0] + theta[1, 0] * xp + theta[2, 0]*xp**2

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, linewidth=1, label='Linear')
plt.title('Linear Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 이제 차원을 더 많이 늘려주자 (9차항까지 )
A = np.hstack([x**i for i in range(10)])
A = np.asmatrix(A)

# optimal Theta 구하기
theta = (A.T*A).I*(A.T*y)

xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)

polybasis = np.hstack([xp**i for i in range(10)])
polybasis = np.asmatrix(polybasis)

yp = polybasis * theta

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, linewidth=1, label='Linear')
plt.title('Linear Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 차원끼리의 비교
d = [1, 3, 5, 9]

# RSS 를 계산해보자 ( 잔차 제곱합 )
RSS = []

for k in range(len(d)):
    A = np.hstack([x**i for i in range(d[k]+1)])
    polybasis = np.hstack([xp**i for i in range(d[k]+1)])

    A = np.asmatrix(A)
    polybasis = np.asmatrix(polybasis)

    theta = (A.T*A).I*(A.T*y)
    yp = polybasis * theta

    # RSS는 2 norm을 실제 제곱하는 과정
    RSS.append(np.linalg.norm(y - A*theta, 2)**2)

    plt.plot(x, y, 'o', label='Data')
    plt.plot(xp, yp)
    plt.title('degree = {}'.format(d[k]))
    plt.grid(alpha=0.3)
    plt.show()

# RSS
plt.figure(figsize=(10, 8))
plt.stem(d, RSS, label='RSS')
plt.title('RSS', fontsize=15)
plt.xlabel('degree', fontsize=15)
plt.ylabel('RSS', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()  # 점차 작아지는 것을 확인할 수 있다

# 2. Overfitting with RBF Functions
xp = np.arange(-4.5, 4.5, 0.01).reshape(-1, 1)

# 10개의 center
d = 10
u = np.linspace(-4.5, 4.5, d)
sigma = 1  # sigma를 올리면, smoothing 되는 효과를 확인할 수 있다

# xp :
rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.asmatrix(rbfbasis)

# X : 실제 데이터들
A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
A = np.asmatrix(A)

theta = (A.T*A).I*(A.T*y)
yp = rbfbasis*theta

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Overfitted with RBF')
plt.title('RSS', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 마찬가지로 차원을 확장할 수록 더 정확해지는 것을 확인할 수 있다
d = [2, 4, 6, 10]
sigma = 1


for k in range(4):
    u = np.linspace(-4.5, 4.5, d[k])

    # X : 실제 데이터들
    A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d[k])])
    rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2))
                          for i in range(d[k])])

    A = np.asmatrix(A)
    rbfbasis = np.asmatrix(rbfbasis)

    theta = (A.T*A).I*(A.T*y)
    yp = rbfbasis*theta

    plt.plot(x, y, 'o', label='Data')
    plt.plot(xp, yp)
    plt.title('num RBFs = {}'.format(d[k]), fontsize=15)
    plt.grid(alpha=0.3)
    plt.show()


# 3. Regularization

# Lasso : ! Norm
# Ridge : 2 Norm

# CVSPY code
# overfitting

# 10개의 center
d = 10
u = np.linspace(-4.5, 4.5, d)
sigma = 1  # sigma를 올리면, smoothing 되는 효과를 확인할 수 있다

# xp :
rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.asmatrix(rbfbasis)

# X : 실제 데이터들
A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
A = np.asmatrix(A)

theta = cvx.Variable((d, 1))
obj = cvx.Minimize(cvx.sum_squares(A*theta-y))  # 우리가 찾고자 하는 것
prob = cvx.Problem(obj, []).solve()

yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Overfitted with RBF')
plt.title('RSS', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# CVSPY code --> Regularization
# ridge regression
lamb = 0.1  # lambda를 키우면, 더 smooth 해진다

theta = cvx.Variable((d, 1))
obj = cvx.Minimize(cvx.sum_squares(A*theta-y) + lamb *
                   cvx.sum_squares(theta))  # 우리가 찾고자 하는 것
prob = cvx.Problem(obj, []).solve()

yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Overfitted with RBF')
plt.title('RSS', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

# 쎄타들 그려보기
plt.figure(figsize=(10, 8))
plt.title('Ridge magnitude of $\theta$', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.stem(np.linspace(1, 10, 10).reshape(-1, 1), theta.value)
plt.xlim([0.5, 10.5])
plt.ylim([-5, 5])
plt.grid(alpha=0.3)
plt.show()

# 주목할 점은 모든 쎄타가 1 ~ -1 사이
# 하지만, 0 이 되는 것은 없다.

lamb = np.arange(0, 3, 0.01)
theta_record = []

for k in lamb:
    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) + k*cvx.sum_squares(theta))
    prob = cvx.Problem(obj).solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize=(10, 8))
plt.plot(lamb, theta_record, linewidth=1)
plt.title('Ridge magnitude as afunction of regularization', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.show()

# 10개의 쎄타가, 각갂의 그래프 형태로 그려져 있는 것
# lamda가 커질 수록 0에 가까워지지만, 0은 아니라는 것


# 4. Sparsity for Feature Selection using Lassor

# 대부분 쎄타를 0으로 보내서
# 중요한 변수만을 뽑아내기
# 2 Norm이 아니라 1 norm 사용하기

# Lassor regression
lamb = 2  # 커지면, 마찬가지로 smoothing 효과
theta = cvx.Variable([d, 1])
obj = cvx.Minimize(cvx.sum_squares(A*theta-y) + lamb*cvx.norm(theta, 1))
prob = cvx.Problem(obj).solve()

yp = rbfbasis * theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Lasso')
plt.title('Lassor Regularization', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(10, 8))
plt.title('Lasso magnitude of $\theta$', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.stem(np.arange(1, 11), theta.value)
plt.xlim([0.5, 10.5])
plt.ylim([-5, 5])
plt.grid(alpha=0.3)
plt.show()

# 주목할 점은 모든 쎄타가 1 ~ -1 사이
# 0 이 되는 쎄타가 대부분인 것을 확인할 수 있다
# lamb 를 높이면 높일 수록, 0으로 가는 쎄타가 더 많아진다

lamb = np.arange(0, 3, 0.01)
theta_record = []

for k in lamb:
    theta = cvx.Variable([d, 1])
    obj = cvx.Minimize(cvx.sum_squares(A*theta - y) + k*cvx.norm(theta, 1))
    prob = cvx.Problem(obj).solve()
    theta_record.append(np.ravel(theta.value))

plt.figure(figsize=(10, 8))
plt.plot(lamb, theta_record, linewidth=1)
plt.title('Lassor coefficient as afunction of regularization', fontsize=15)
plt.xlabel(r'$\theta$', fontsize=15)
plt.ylabel('magnitude', fontsize=15)
plt.show()

# 10개의 쎄타가, 각갂의 그래프 형태로 그려져 있는 것
# 0에 도달하는 쎄타가 많은 것을 확인할 수 있다
# non - zero인 feature만 선택하기
# 2,3,8,10번째만 가져오기

d = 4
u = np.array([-3.5, -2.5, 2.5, 4.5])
sigma = 1

rbfbasis = np.hstack([np.exp(-(xp-u[i])**2/(2*sigma**2)) for i in range(d)])
rbfbasis = np.asmatrix(rbfbasis)

A = np.hstack([np.exp(-(x-u[i])**2/(2*sigma**2)) for i in range(d)])
A = np.asmatrix(A)

theta = cvx.Variable((d, 1))
obj = cvx.Minimize(cvx.norm(A*theta-y, 2))  # 우리가 찾고자 하는 것 : 2 Norm
prob = cvx.Problem(obj, []).solve()

yp = rbfbasis*theta.value

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'o', label='Data')
plt.plot(xp, yp, label='Sparse')
plt.title('Regression with Selected Features', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.legend(fontsize=15)
plt.grid(alpha=0.3)
plt.show()
