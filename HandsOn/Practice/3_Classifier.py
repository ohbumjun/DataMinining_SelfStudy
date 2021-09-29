{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "source": [
    "# 파이썬 ≥3.5 필수\r\n",
    "import sys\r\n",
    "assert sys.version_info >= (3, 5)\r\n",
    "\r\n",
    "# 사이킷런 ≥0.20 필수\r\n",
    "import sklearn\r\n",
    "assert sklearn.__version__ >= \"0.20\"\r\n",
    "\r\n",
    "# 공통 모듈 임포트\r\n",
    "import numpy as np\r\n",
    "import os\r\n",
    "\r\n",
    "# 노트북 실행 결과를 동일하게 유지하기 위해\r\n",
    "np.random.seed(42)\r\n",
    "\r\n",
    "# 깔끔한 그래프 출력을 위해\r\n",
    "%matplotlib inline\r\n",
    "import matplotlib as mpl\r\n",
    "import matplotlib.pyplot as plt\r\n",
    "mpl.rc('axes', labelsize=14)\r\n",
    "mpl.rc('xtick', labelsize=12)\r\n",
    "mpl.rc('ytick', labelsize=12)\r\n",
    "\r\n",
    "# 그림을 저장할 위치\r\n",
    "PROJECT_ROOT_DIR = \".\"\r\n",
    "CHAPTER_ID = \"classification\"\r\n",
    "IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, \"images\", CHAPTER_ID)\r\n",
    "os.makedirs(IMAGES_PATH, exist_ok=True)\r\n",
    "\r\n",
    "def save_fig(fig_id, tight_layout=True, fig_extension=\"png\", resolution=300):\r\n",
    "    path = os.path.join(IMAGES_PATH, fig_id + \".\" + fig_extension)\r\n",
    "    print(\"그림 저장:\", fig_id)\r\n",
    "    if tight_layout:\r\n",
    "        plt.tight_layout()\r\n",
    "    plt.savefig(path, format=fig_extension, dpi=resolution)"
   ],
   "outputs": [],
   "metadata": {}
  }
 ],
 "metadata": {
  "orig_nbformat": 4,
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}