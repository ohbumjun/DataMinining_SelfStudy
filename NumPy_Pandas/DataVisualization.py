import datetime
import platform
import matplotlib
from matplotlib import font_manager, rc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 환경 구성
# 시각화 그려주는데, 화면에 보여줄 때, 얼만큼 크게 나타나게 할지
plt.rcParams['figure.figsize'] = [10, 8]  # 가로 10,세로 8
sns.set(style='whitegrid')
sns.set_palette('pastel')  # 시각화 테마
warnings.filterwarnings('ignore')

# -------------------------------
# % 한글이 깨지는 경우

if platform.system() == 'Windows':
    # 윈도우인 경우
    font_name = font_manager.FontProperties(
        fname='c:/Windows/Fonts/malgun.ttf').get_name()

    rc('font', family=font_name)
else:
    # Mac인 경우
    rc('font', family='AppleGothic')

matplotlib.rcParams['axes.unicode_minus'] = False

# Loading 'Tips' dataset from seaborn
tips = sns.load_dataset('tips')
tips.head()

# matplotlib을 활용한 시각화
# 요일별로 총 얼마나 받았는지
sum_tips_by_day = tips.groupby('day')['tip'].sum()
sum_tips_by_day

x_label = ['Thu', 'Fri', 'Sat', 'Sun']
x_label_index = np.arange(len(x_label))  # 0 ~ 3 까지 list 숫자 생성
x_label_index

# Bar 차트 이해 및 제작
plt.bar(x_label, sum_tips_by_day,  # x 축, y 축
        color='pink',  # 색 지정
        alpha=0.6,  # 투명도
        width=0.3,  # 두께
        align='edge'  # 왼쪽 구석으로 붙는다
        )
# 제목
plt.title('Sum of Tips by Day')
# x 축 라벨링
plt.xlabel('Days', fontsize=14)
plt.ylabel('Sum of Tips', fontsize=14)
# x 축 라벨링 조정하기
plt.xticks(x_label_index,
           x_label,
           rotation=45,
           fontsize=15
           )
plt.show()


# seaborn을 활용한 시각화
sns.barplot(data=tips,  # 데이터 프레임 명시
            x='day',
            y='tip',
            estimator=np.sum,  # 요일별 합의 데이터
            hue='sex',  # 성별 기준으로 보고 싶다
            palette='pastel',
            order=['Sun', 'Sat', 'Fri', 'Thur'],  # x ticks 순서
            edgecolor='.6',  # 테두리 선명도
            linewidth=2.5  # 테두기 두께
            )
plt.title('Sum Of Tips by Days', fontsize=16)
plt.xlabel('Days')
plt.ylabel('Sum Of Tips')

plt.xticks(rotation=45)

plt.show()

# Pie Chart 이해 및 제작하기
# matplotlib을 활용한 시각화
sum_tips_by_day = tips.groupby('day')['tip'].sum()
ratio_tip_by_day = sum_tips_by_day/sum_tips_by_day.sum()

x_label = ['Thu', 'Fri', 'Sat', 'Sun']
plt.pie(ratio_tip_by_day,  # 비율 값
        labels=x_label,  # 라벨 값
        autopct='%.1f%%',  # 부채꼴 안에 표시될 숫자 형식
        startangle=90,  # 축이 시작되는 각도
        counterclock=True,  # 시계방향순 표시
        explode=[0.05, 0.05, 0.05, 0.05],  # 떨어뜨려서 보기 (중심에서 벗어난 정도)
        shadow=True,  # 그림자 표시 여부
        colors=['gold', 'silver', 'whitesmoke', 'gray'],
        # 안에 도넛 모양 주기
        wedgeprops={'width': 0.8, 'edgecolor': 'white', 'linewidth': 3}
        )


plt.show()

# Line char 이해 및 제작
# matplotlib을 활용해서 시각화

# line 차트 예제를 위해, tips 데이터에 가상 시간 컬럼 추가하기
# 일요일 데이터만 사용하기
sun_tips = tips[tips['day'] == 'Sun']
sun_tips

# 현재 서버 시간을 얻기 위해 datetime 라이브러리 사용
date = []
today = datetime.date.today()
date.append(today)

# 가상으로 date 데이터 만들기
# sun_tips.shape[0] : 총 row 수
for i in range(sun_tips.shape[0]-1):
    today += datetime.timedelta(1)  # 하루씩 추가
    date.append(today)
date

sun_tips['date'] = date
sun_tips

# line chart
plt.plot(sun_tips['date'], sun_tips['total_bill'],
         linestyle='-',  # --  : 점선/ --/ - / -. 등등
         linewidth=2,
         color='pink',
         alpha=1,  # 투명도
         )
plt.title('Total Tips by Date', fontsize=20)
plt.xticks(rotation=90)
plt.show()

# seaborn을 활용한 line char 시각화
sns.lineplot(
    data=sun_tips,
    x='date',
    y='total_bill',
    hue='sex',  # 성별 기준 분리
    palette='pastel'
)
plt.title('Total Bill By date & sex')

plt.show()

# Scatter 차트 이해 및 제작
# matplotlib을 활용한 시각화
plt.scatter(tips['total_bill'], tips['tip'],
            color='pink',
            edgecolor='black',  # 테두리 색상
            linewidth=2
            )
plt.show()

# seaborn을 활용한 시각화
sns.scatterplot(data=tips,
                x='total_bill',
                y='tip',
                style='time',  # style : 모양 구분으로 다른 변수랑 구분
                hue='day',  # hue : 요일별 다른 색상
                size='size'  # size : 크기(변수 중에 순서척도 같은 경우
                # ex) 만족도 조사
                # ex) 1 ~ 5 --> 만족도 별로 동그라미 표시)로 구분
                )
plt.title('Scatter between total_bill and tip', fontsize=20)
plt.xlabel('total_bill', fontsize=16)
plt.ylabel('tip', fontsize=12)
plt.show()


# -----------------------------
# Heat Map 차트 이해 및 목적
tips.corr()  # 상관관계 : 연속형 변수간의 상관관계

# seaborn 을 활용한 시각화
sns.heatmap(tips.corr(),
            annot=True,
            square=True,  # 정사각형으로 만들기
            vmin=-1, vmax=1,  # 범위지정 --> default : 0 ~ 1
            linewidth=.5,
            cmap='RdBu',  # 색상 테마
            )

plt.show()

# Histogram 차트 이해 및 제작
# Histogram ? : 해당 column에 대한 각 변수들이 몇개 있는지 확인해보기 위함
# matplotlib을 활용한 시각화
plt.hist(tips['total_bill'],
         bins=30,  # 잘게 잘게 x 변수를 쪼개서 볼지 말지
         density=True,  # False : y축은 개수 / True : y축은 비율
         alpha=0.7,  # 선명도
         color='pink',
         edgecolor='black',
         linewidth=0.9,
         )
plt.title('Histogram for total_bill')
plt.xlabel('total_bill')
plt.ylabel('rate')
plt.show()


# seaborn을 활용한 시각화
sns.histplot(data=tips,
             x='total_bill',  # y축은 필요 x
             bins=30,
             kde=True,  # kernel density estimate : 히스토그램 분포도를 그림
             hue='sex',
             multiple='stack',  # hue를 쓸 때, 따로따로 구분해서 보이게 하기
             shrink=0.8,  # 그래프의 두께 조절
             )
plt.title('Histogram for total_bill')
plt.xlabel('total_bill')
plt.ylabel('rate')
plt.show()

# ------------------------------
# Box Chart 차트 이해 및 제작
# Box Chart란
# 한 변수에 대해, median,min,max, outlier , IQR(25 ~ 75% 사이)

# matplotlib을 활용한 시각화
plt.boxplot(tips['tip'],
            sym='rs'  # outlier 를 어떻게 표시할 것인가
            )
plt.title('Box Plot for Tip', fontsize=20)
plt.xlabel('tip', fontsize=15)
plt.ylabel('tip size', fontsize=15)
plt.show()

# seaborn을 활용한 시각화
sns.boxplot(data=tips,
            x='day',  # 요일별 tip
            y='tip',
            hue='smoker',  # 흡연자들 별, 팁주는 정도
            linewidth=3,
            order=['Sun', 'Sat', 'Fri', 'Thur']
            )
plt.title('Box Plot for Tip', fontsize=20)
plt.xlabel('day', fontsize=15)
plt.ylabel('tip', fontsize=15)
plt.show()
