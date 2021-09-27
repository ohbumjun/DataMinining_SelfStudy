# 1. HandWritten Digit Classification
# input : 손으로 쓰여진 이미지
# ouput : 그것이 0이냐, 1이냐
# 이를 구별하기 위해 classifire를 사용
# 중간에 hand-engineered feature들을 사용
# input -->hand-engineered feature --> classifier --> output

# input 중에서, classifier에 적용하기 위해서
# 특징 인자를 뽑아내야 한다
# 즉, 많은 feature 중에서 classify가 잘 되게 하는
# feature를 선별해야 한다

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
%matplotlib inline

# generate 3 simulated clusters
# 각 집단의 평균
mu1 = np.array([1, 7])
mu2 = np.array([3, 4])
mu3 = np.array([6, 5])

# Cov matrix
SIGMA1 = 0.8 * np.array([[1, 1.5],
                         [1.5, 3]])
SIGMA2 = 0.5 * np.array([[2, 0],
                         [0, 2]])
SIGMA3 = 0.5 * np.array([[1, -1],
                         [-1, 2]])
# 2차원 정규분포
# ex) mu1 = 평균, cov matrix, sample 개수
X1 = np.random.multivariate_normal(mu1, SIGMA1, 100)
X2 = np.random.multivariate_normal(mu2, SIGMA2, 100)
X3 = np.random.multivariate_normal(mu3, SIGMA3, 100)

# 각 집단에 대한 y label
y1 = 1*np.ones([100, 1])  # label 1
y2 = 2*np.ones([100, 1])  # label 2
y3 = 3*np.ones([100, 1])  # label 3

plt.figure(figsize=(10, 8))
plt.title("Generated Data", fontsize=15)
plt.plot(X1[:, 0], X1[:, 1], '.', label='C1')
plt.plot(X2[:, 0], X2[:, 1], '.', label='C2')
plt.plot(X3[:, 0], X3[:, 1], '.', label='C3')

plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.legend(fontsize=12)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.axis([-2, 10, 0, 12])
plt.show()

# np.ravel : 다차원 배열을 1차원 배열로 평평하게 해주는 것
clf_12 = linear_model.LogisticRegression(solver='lbfgs')
clf_13 = linear_model.LogisticRegression(solver='lbfgs')
clf_23 = linear_model.LogisticRegression(solver='lbfgs')

clf_12.fit(np.vstack([X1, X2]), np.ravel(np.vstack([y1, y2])))
clf_13.fit(np.vstack([X1, X3]), np.ravel(np.vstack([y1, y3])))
clf_23.fit(np.vstack([X2, X3]), np.ravel(np.vstack([y2, y3])))

w_12 = np.zeros([3, 1])
w_12[0, 0] = clf_12.intercept_[0]
w_12[1, 0] = clf_12.coef_[0, 0]
w_12[2, 0] = clf_12.coef_[0, 1]

w_13 = np.zeros([3, 1])
w_13[0, 0] = clf_13.intercept_[0]
w_13[1, 0] = clf_13.coef_[0, 0]
w_13[2, 0] = clf_13.coef_[0, 1]

w_23 = np.zeros([3, 1])
w_23[0, 0] = clf_23.intercept_[0]
w_23[1, 0] = clf_23.coef_[0, 0]
w_23[2, 0] = clf_23.coef_[0, 1]

xlp = np.linspace(-2, 10, 100).reshape(-1, 1)
x2p_12 = -w_12[1, 0]/w_12[2, 0] * xlp - w_12[0, 0]/w_12[2, 0]
x2p_13 = -w_13[1, 0]/w_13[2, 0] * xlp - w_13[0, 0]/w_13[2, 0]
x2p_23 = -w_23[1, 0]/w_23[2, 0] * xlp - w_23[0, 0]/w_23[2, 0]


plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], '.', label='C1')
plt.plot(X2[:, 0], X2[:, 1], '.', label='C2')
plt.plot(X3[:, 0], X3[:, 1], '.', label='C3')
plt.plot(xlp, x2p_12, '--', linewidth=3, label='1 vs 2')
plt.plot(xlp, x2p_13, '--', linewidth=3, label='1 vs 3')
plt.plot(xlp, x2p_23, '--', linewidth=3, label='2 vs 3')
plt.axis('equal')
plt.xlabel('feature 1', fontsize=15)
plt.ylabel('feature 2', fontsize=15)
plt.legend(fontsize=12)
plt.show()

# 각각의 데이터에 대해서
# 3개의 binary class 비교집단에 대해, 각 집단에 속할 확률을 비교
# 그 중에서 , 가장 확률이 큰 집단에 속한다고 결론 내리면 된다
test_x = [[0, 8]]
isClass01 = clf_12.predict(test_x)[0] == 1 and clf_13.predict(test_x)[0] == 1
isClass02 = clf_12.predict(test_x)[0] == 2 and clf_23.predict(test_x)[0] == 2
isClass03 = clf_13.predict(test_x)[0] == 3 and clf_23.predict(test_x)[0] == 3


def class_check(test_x):
      isClass01 = clf_12.predict(
          test_x)[0] == 1 and clf_13.predict(test_x)[0] == 1
  isClass02 = clf_12.predict(test_x)[0] == 2 and clf_23.predict(test_x)[0] == 2
  isClass03 = clf_13.predict(test_x)[0] == 3 and clf_23.predict(test_x)[0] == 3

  if isClass01 : return 1
  elif isClass02 : return 2
  elif isClass03 : return 3
  return 0

# 각 영역에 해당하는 색상 칠하기
[X1gr,X2gr] = np.meshgrid(np.arange(-2,10,0.3), np.arange(0,12,0.3))
Xp = np.hstack([X1gr.reshape(-1,1),X2gr.reshape(-1,1)])
Xp = np.asmatrix(Xp) # test data set
Xp.shape

plt.figure(figsize = (10,8))
plt.plot(X1[:,0],X1[:,1],'.',label='C1')
plt.plot(X2[:,0],X2[:,1],'.',label='C2')
plt.plot(X3[:,0],X3[:,1],'.',label='C3')
plt.axis('equal')
plt.axis([-2,10,0,12])
plt.xlabel('feature 1',fontsize = 15)
plt.ylabel('feature 2',fontsize = 15)
plt.legend(fontsize = 12)

# 각 class가 어느 test
for test_x in Xp :
  g = class_check(test_x) 
  if g == 1 :
    c = 'blue'
  elif g == 2 :
    c = 'orange'
  elif g == 3 :
    c = 'green'
  else :
    c = 'white'
  plt.plot(test_x[0,0],test_x[0,1],'s',alpha = 0.3, color = c, markersize = 8)

plt.show()
