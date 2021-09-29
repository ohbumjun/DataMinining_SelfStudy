# 파이썬 ≥3.5 필수
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import sklearn
import sys
assert sys.version_info >= (3, 5)

# 사이킷런 ≥0.20 필수
assert sklearn.__version__ >= "0.20"

# 공통 모듈 임포트

# 노트북 실행 결과를 동일하게 유지하기 위해
np.random.seed(42)

# 깔끔한 그래프 출력을 위해
%matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 그림을 저장할 위치
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("그림 저장:", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# 미국 고등학생 및 인구조사국 직원들이 쓴 70,000개의
# 작은 숫자 이미지를 모은 MNIST 데이터셋
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()


X, y = mnist["data"], mnist["target"]
X.shape
# 70000개의 이미지, 각 이미지는 784개의 특성을 지니고 있다
# 왜냐하면 하나의 이미지가 28*28 px 이기 때문이다
# 개개의 특성은, 0(흰색) 부터 255(검은색) 까지의 픽셀 강도를 나타낸다


%matplotlib inline

# 샘플의 특성 벡터 추출
some_digit = X[0]
# 28 * 28 배열로 바꾸기
some_digit_image = some_digit.reshape(28, 28)
