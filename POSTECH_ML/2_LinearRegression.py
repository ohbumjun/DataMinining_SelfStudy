# 1. Linear Regression
# 1_1 Linear Algebra

from sklearn import linear_model
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# data points in column vector ( 13개 : input - output)
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8,
              3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8,
              2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko')
plt.title('Data', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()


m = x.shape[0]
A = np.hstack([np.ones([m, 1]), x])
# A = np.hstack([x**0,x])
A

A = np.asmatrix(A)
theta1 = (A.T*A).I*(A.T*y)
print("theta : ", theta)

# datas
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

# to plot a straight line (fiteed line)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp = theta1[0, 0] + theta1[1, 0] * xp

plt.plot(xp, yp, 'r', linewidth=2, label="regression")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

# 1_2) CVXPY Optimization
theta2 = cvx.Variable((2, 1))
obj = cvx.Minimize(cvx.norm(A*theta2 - y, 2))
prob = cvx.Problem(obj, [])
result = prob.solve()

print("theta : \n", theta2.value)

theta1 = cvx.Variable((2, 1))
# theta2는 2 Norm, theta2은 1 Norm
obj = cvx.Minimize(cvx.norm(A*theta1 - y, 1))
prob = cvx.Problem(obj, [])
result = prob.solve()

print("theta : \n", theta1.value)

# datas
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

# to plot a straight line (fiteed line)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp1 = theta1.value[0, 0] + theta1.value[1, 0] * xp
yp2 = theta2.value[0, 0] + theta2.value[1, 0] * xp

plt.plot(xp, yp1, 'r', linewidth=2, label="$L_1$")
plt.plot(xp, yp2, 'b', linewidth=2, label="$L_2$")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

# 1_3) Outlier
# add outlier  --> np.vstack : 두 배열을 위에서 아래로 붙이기
x = np.vstack([x, np.array([0.5, 3.8]).reshape(-1, 1)])
y = np.vstack([y, np.array([3.9, 0.3]).reshape(-1, 1)])

A = np.hstack([x**0, x])
A = np.asmatrix(A)

# datas
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label="data")
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.grid(alpha=0.3)
plt.show()

theta2 = cvx.Variable((2, 1))
obj2 = cvx.Minimize(cvx.norm(A*theta2 - y, 2))
prob2 = cvx.Problem(obj2, [])
result = prob.solve()

theta2 = cvx.Variable((2, 1))
# 2 Norm
obj2 = cvx.Minimize(cvx.norm(A*theta2 - y, 2))
prob2 = cvx.Problem(obj2).solve()

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')

xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp2 = theta2.value[0, 0] + theta2.value[1, 0] * xp

# 기존의 그래프와 차이가 있는 것을 확인할 수 있다
# norm2 의 경우, outlier가 존재하면, 정확도가 줄어든다
plt.plot(xp, yp2, 'r', linewidth=2, label="$L_1$")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

# 1 Norm
theta1 = cvx.Variable((2, 1))
obj1 = cvx.Minimize(cvx.norm(A*theta1 - y, 1))
prob1 = cvx.Problem(obj1).solve()

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ko', label='data')

xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp1 = theta1.value[0, 0] + theta1.value[1, 0] * xp

# 기존의 그래프와 차이가 있는 것을 확인할 수 있다
# norm2 의 경우, outlier가 존재하면, 정확도가 줄어든다
plt.plot(xp, yp1, 'r', linewidth=2, label="$L_1$")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

# 같이 그리기
# datas
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

# to plot a straight line (fiteed line)
xp = np.arange(0, 5, 0.01).reshape(-1, 1)
yp1 = theta1.value[0, 0] + theta1.value[1, 0] * xp
yp2 = theta2.value[0, 0] + theta2.value[1, 0] * xp

plt.plot(xp, yp1, 'r', linewidth=2, label="$L_1$")
plt.plot(xp, yp2, 'b', linewidth=2, label="$L_2$")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()

# 1_4) Skit Learn
x = np.array([0.1, 0.4, 0.7, 1.2, 1.3, 1.7, 2.2, 2.8,
              3.0, 4.0, 4.3, 4.4, 4.9]).reshape(-1, 1)
y = np.array([0.5, 0.9, 1.1, 1.5, 1.5, 2.0, 2.2, 2.8,
              2.7, 3.0, 3.5, 3.7, 3.9]).reshape(-1, 1)
##
reg = linear_model.LinearRegression()
reg.fit(x, y)

reg.coef_  # 쎄타 1 ( 기울기 )
reg.intercept_  # 쎄타 0 ( 절편 )


# 같이 그리기
# datas
plt.figure(figsize=(10, 8))
plt.title('Regression', fontsize=15)
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.plot(x, y, 'ko', label="data")

plt.plot(xp, reg.predict(xp), 'r', linewidth=2, label="regression")
plt.legend(fontsize=15)
plt.axis("equal")
plt.grid(alpha=0.3)
plt.xlim([0, 5])
plt.show()
