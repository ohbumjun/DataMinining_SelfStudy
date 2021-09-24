# 최적화 Optimization
# 1) Linear Programming

import numpy as np
import cvxpy as cvx

f = np.array([[3], [3/2]])
lb = np.array([[-1], [0]])
ub = np.array([[2], [3]])

x = cvx.Variable((2, 1))

obj = cvx.Minimize(-f.T*x)
constraints = [lb <= x, x <= ub]

prob = cvx.Problem(obj, constraints)
result = prob.solve()
print(x.value)  # 그#ㅐ의 x값
print(result)  # 약 -10.5

# 2. Quadratic Programming
f = np.array([[3], [4]])
H = np.array([[1/2, 0], [0, 0]])

A = np.array([[-1, -3], [2, 5], [3, 4]])
B = np.array([[-15], [100], [80]])
lb = np.array([[0], [0]])

x = cvx.Variable((2, 1))

obj = cvx.Minimize(cvx.QuadForm(x, H) + f.T*x)
constraints = [A*x <= B, lb <= x]

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)

# 가수까지의 거리 예제
f = np.array([[6], [6]])
H = np.array([[1, 0],
              [0, 1]])

A = np.array([[1, 1]])
B = np.array([[3]])
lb = np.array([[0], [0]])

x = cvx.Variable((2, 1))

obj = cvx.Minimize(cvx.QuadForm(x, H) - f.T*x)
constraints = [A*x <= B, lb <= x]

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)

# 물통까지의 최소 거리
# a,b의 좌표점
a = np.array([[0], [1]])
b = np.array([[4], [2]])

# x2가 0인 제약조건
Aeq = np.array([0, -1])
Beq = np.array(0)

x = cvx.Variable((2, 1))

# 2 ? : Norm을 구하는 과정에서 2Nrom을 사용하겠다
obj = cvx.Minimize(cvx.norm(a-x, 2) + cvx.norm(b-x, 2))
constraints = [Aeq*x == Beq]  # x2가 0이라는 constraints

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(x.value)
print(result)  # 5

# 물류문제 : 3지점까지의 거리 최소값
a = np.array([[np.sqrt(3)], [0]])
b = np.array([[-np.sqrt(3)], [0]])
c = np.array([[0], [3]])

x = cvx.Variable((2, 1))

obj = cvx.Minimize(cvx.norm(a-x, 2) + cvx.norm(b-x, 2) + cvx.norm(c-x, 2))

prob = cvx.Problem(obj)
result = prob.solve()

print(x.value)
print(result)

# Gradient Descent
H = np.matrix([[2, 0], [0, 2]])
g = -np.matrix([[6], [6]])

# 초기화 : (0,0) 에서 시작
x = np.zeros((2, 1))
alpha = 0.2

# gradient 값
for _ in range(25):
    df = H*x + g
    x = x - alpha*df

print(x)  # 3에 가까워진 것을 확인할 수 있다
