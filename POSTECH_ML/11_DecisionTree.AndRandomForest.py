# Decision Tree
# 구조가 tree 구조
# 각각의 노드에서 결정을 내린다.

# - Not Numerice
# - 모든 feature가 중요하지 않을 수 있다
# - 일부 feature만이 중요하고
# - cost 라는 것이 존재한다

'''
좋은 tree 라는 것은 무엇일까 ?
- small tree ! 
- 각 Feature 에서, 각각의 value 기준에 대한 disorder를 구하고
- 각 Feature 에서의 Quality Test 를 진행한다
- 그 결과 Quality Test가 낮을 수록 좋은 Feature라고 할 수 있다
- 그만큼 확실하게 구분해준다는 것이기 때문이다

Decision Tree의 장점은,
모든 종류의 Test를 진행하지 않는다는 것이다

제일 좋은 Feature 를 가지고만
해당 데이터들을 test한다라는 장점이 있다 

'''

# Disorder 그래프
from sklearn import ensemble
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.linspace(0, 1, 100)
D = -x*np.log2(x) - (1-x)*np.log2(1-x)

plt.figure(figsize=(10, 8))
plt.plot(x, D, linewidth=3)
plt.xlabel(r'$x$', fontsize=15)
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# Quality test


def D(x):
    y = -x*np.log2(x) - (1-x)*np.log2(1-x)
    return


data = np.array([[0, 0, 1, 0, 0],
                 [1, 0, 2, 0, 0],
                 [0, 1, 2, 0, 1],
                 [0, 1, 0, 1, 1],
                 [1, 1, 1, 2, 0],
                 [1, 1, 0, 2, 0],
                 [0, 0, 2, 1, 0]])
x = data[:, 0:4]  # 앞에 4열은 x
y = data[:, 4]  # 맨 마지막 열은 y
print(y)

# clf
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=0)
clf.fit(x, y)
clf.predict([[0, 1, 0, 1]])

# Nonlinear Classfication
# 2개의 class 나누기
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


N = X1.shape[0]
M = X0.shape[0]

X = np.vstack([X1, X0])
y = np.vstack([np.ones([N, 1]), np.zeros([M, 1])])

clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=4,
                                  random_state=0)
clf.fit(X, np.ravel(y))

clf.predict([[0, 0]])

# to plot
[X1gr, X2gr] = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

q = clf.predict(Xp)
q = np.asmatrix(q).reshape(-1, 1)

C1 = np.where(q == 1)[0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], 'ro', label='C!')
plt.plot(X0[:, 0], X0[:, 1], 'bo', label='C0')
plt.plot(Xp[C1, 0], Xp[C1, 1], 'gs', markersize=10, alpha=0.1, label='SVM')
plt.title('SVM for Kernel', fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.show()

# 아래의 색칠해진 영역은
# max_depth가 커짐에 따라 작아지는 것을 알 수 있다
# max_depth가 2라는 것은, 1번, 2번만 나눈 것


# Multi class

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


X = np.vstack([X1, X2, X3])
y = np.vstack([y1, y2, y3])

clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  max_depth=2,
                                  random_state=0)
clf.fit(X, np.ravel(y))

# to plot
res = 0.3
[X1gr, X2gr] = np.meshgrid(np.arange(-2, 10, res),
                           np.arange(0, 12, res))
Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

q = clf.predict(Xp)
q = np.asmatrix(q).reshape(-1, 1)

C1 = np.where(q == 1)[0]
C2 = np.where(q == 2)[0]
C3 = np.where(q == 3)[0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], '.', label='C1')
plt.plot(X2[:, 0], X2[:, 1], '.', label='C2')
plt.plot(X3[:, 0], X3[:, 1], '.', label='C3')
plt.plot(Xp[C1, 0], Xp[C1, 1], 's', color='blue', markersize=8, alpha=0.1)
plt.plot(Xp[C2, 0], Xp[C2, 1], 's', color='orange', markersize=8, alpha=0.1)
plt.plot(Xp[C3, 0], Xp[C3, 1], 's', color='green', markersize=8, alpha=0.1)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.grid(alpha=0.3)
plt.axis([-2, 10, 0, 12])
plt.show()

# Random Forest

clf = ensemble.RandomForestClassifier(
    criterion='entropy',
    n_estimators=100,
    max_depth=3,
    random_state=0
)
clf.fit(X, np.ravel(y))

# to plot
res = 0.3
[X1gr, X2gr] = np.meshgrid(np.arange(-2, 10, res),
                           np.arange(0, 12, res))
Xp = np.hstack([X1gr.reshape(-1, 1), X2gr.reshape(-1, 1)])
Xp = np.asmatrix(Xp)

q = clf.predict(Xp)
q = np.asmatrix(q).reshape(-1, 1)

C1 = np.where(q == 1)[0]
C2 = np.where(q == 2)[0]
C3 = np.where(q == 3)[0]

plt.figure(figsize=(10, 8))
plt.plot(X1[:, 0], X1[:, 1], '.', label='C1')
plt.plot(X2[:, 0], X2[:, 1], '.', label='C2')
plt.plot(X3[:, 0], X3[:, 1], '.', label='C3')
plt.plot(Xp[C1, 0], Xp[C1, 1], 's', color='blue', markersize=8, alpha=0.1)
plt.plot(Xp[C2, 0], Xp[C2, 1], 's', color='orange', markersize=8, alpha=0.1)
plt.plot(Xp[C3, 0], Xp[C3, 1], 's', color='green', markersize=8, alpha=0.1)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.grid(alpha=0.3)
plt.axis([-2, 10, 0, 12])
plt.show()

# 조금 더 좋은 결과가 나온 것을
# 확인할 수 있다
