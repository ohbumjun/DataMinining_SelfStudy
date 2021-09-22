# 1. Linear Algebra
# 1_1 Vector Define , Inverse Matrix

import numpy as np

A = np.array([[4, 5], [-2, 3]])
print(A)

B = np.array([[-13], [9]])
print(B)

# inverse
np.linalg.inv(A)

np.linalg.inv(A).dot(B)

# 연산을 조금 더 직관적으로 수행하기 위해서
# 행렬의 크기가 크지 않다면 asmatrix 함수를 사용한다
A = np.asmatrix(A)
B = np.asmatrix(B)
A.I*B

# 1_2) Vector-Vector Products
x = np.array([[1], [1]])
y = np.array([[2], [3]])
x.T.dot(y)

x = np.asmatrix(x)
y = np.asmatrix(y)
x.T*y

z = x.T*y
print(z)

# 1_3) Norm

# norm : 벡터에서의 크기
x = np.array([[4], [3]])
print(x)
np.linalg.norm(x, 2)  # 2 : 2 norm으로 정의한다

# 1_4) Orthogonality
x = np.matrix([[1], [2]])
y = np.matrix([[2], [-1]])
x.T*y  # 0이므로 수직임을 알 수 잇다
