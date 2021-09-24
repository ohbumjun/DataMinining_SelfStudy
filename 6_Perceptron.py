# 1. Classification

# h(x) = sgin(w.T*x)
# -1 혹은 1

# g(x) == 0 인 hyper plane을 찾는 과정

# g(x)==0 은 boundary 이고, h는 임의의 점 x와 g(x) == 0 선과의 직선 거리
# g(x) > 0 이라는 것은, h > 0 ( 양수 쪽에 있다는 것 )
# 그반대도 성립

# Perceptron 알고리즘
# sign(W.T*x)와 실제 label yn이 서로 다를 경우만,
# w = w + yn*xn 이라는 rule을 통해 update를 한다

# input x1...xn이 주어지면
# 임의의 w 벡터를 정하고, 실제 label과 predicted된 label이 다르다면
# update를 진행한다

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# training data generation
m = 100
x1 = 8*np.random.rand(m, 1)  # 1부터 8 사이의 임의 데이터 100행 1열 벡터
x2 = 7*np.random.rand(m, 1)-4

g = 0.8*x1 * x2 - 3
print(g)

# np.where : 조건에 맞는 idx를 반환한다
C1 = np.where(g >= 1)
C0 = np.where(g < -1)
print(C1)  # Class를 C0, C1로 나눈다

C1 = np.where(g >= 1)[0]
C0 = np.where(g < -1)[0]
print(C1.shape)
print(C0.shape)

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.title('Linearly Seperable Classes', fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.show()

N = C1.shape[0]  # g>=1 인 애들의 개수
M = C0.shape[0]  # g < -1 인 애들의 개수

print(N)
print(M)

# X1,X2 중에서 C1에 있는 것들만
X1 = np.hstack([np.ones([N, 1]), x1[C1], x2[C1]])
# X1,X2 중에서 C0에 있는 것들만
X0 = np.hstack([np.ones([M, 1]), x1[C0], x2[C0]])

# 위,아래로 쌓는다
X = np.vstack([X1, X0])
# 처음 N행은, 1, 그 다음 M행은 -1로 label 한다
Y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

X = np.asmatrix(X)
y = np.asmatrix(Y)

w = np.zeros([3, 1])  # 처음은 임의로 선택 [0,0,0]
w = np.asmatrix(w)
print(w)

n_iter = N+M  # total 개수

# misclassified되는 것이 한번도 없을 때까지 반복
'''
while True : 
  misClassified = False
  # 모든 데이터에 대해서 1번 돈 것
  for i in range(n_iter):
    # miss classified 되었을 때
    if y[i,0] != np.sign(X[i,:]*w)[0,0] : # 첫번째 element 가져오기 
      w = w + y[i,0] * X[i,:].T
      misClassified = True 
  if not misClassified : break
print(w)
'''

# 간단하게 100번만 돌리기
for _ in range(100):
    # 모든 데이터에 대해서 1번 돈 것
    for i in range(n_iter):
        # miss classified 되었을 때
        if y[i, 0] != np.sign(X[i, :]*w)[0, 0]:  # 첫번째 element 가져오기
            w = w + y[i, 0] * X[i, :].T
print(w)

# g(x) = w0 + w.T*x = w0 + w1*x1 + w2*x2 = 0
# x2 = -(w1//w2)*x1 - w0//w1

xlp = np.linspace(0, 8, 100).reshape(-1, 1)  # 임의의 데이터들
# 직선을 그릴 것이다 --> 임의의 데이터들의 x2 값
x2p = -(w[1, 0]/w[2, 0])*xlp - w[0, 0]/w[2, 0]

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xlp, x2p, c='k', linewidth=3, label='perceptron')
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.show()

# Sckit-Learn 사용하기
# X1,X2 중에서 C1에 있는 것들만
# 여기서는 x1차, x2차만 필요하다
X1 = np.hstack([x1[C1], x2[C1]])
# X1,X2 중에서 C0에 있는 것들만
X0 = np.hstack([x1[C0], x2[C0]])

# 위,아래로 쌓는다
X = np.vstack([X1, X0])
# 처음 N행은, 1, 그 다음 M행은 -1로 label 한다
y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])


clf = linear_model.Perceptron()
clf.fit(X, y)

# boundary가 clf에 저장

clf.predict([[3, -2]])  # -1로 예측
clf.coef_  # w벡터의 값들 : w1,w2
clf.intercept_  # w0

w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]

# g(x) = w0 + w.T*x = w0 + w1*x1 + w2*x2 = 0
# x2 = -(w1//w2)*x1 - w0//w1
xlp = np.linspace(0, 8, 100).reshape(-1, 1)  # 임의의 데이터들
# 직선을 그릴 것이다 --> 임의의 데이터들의 x2 값
x2p = -(w1/w2)*xlp - w0/w2

plt.figure(figsize=(10, 8))
plt.plot(x1[C1], x2[C1], 'ro', alpha=0.4, label='C1')
plt.plot(x1[C0], x2[C0], 'bo', alpha=0.4, label='C0')
plt.plot(xlp, x2p, c='k', linewidth=3, label='perceptron')
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.show()
