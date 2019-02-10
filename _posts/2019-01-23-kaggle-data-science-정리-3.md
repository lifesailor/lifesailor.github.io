---
title: How to Win a Data Science Competition - Week 4 - Tips and Tricks
categories:
  - kaggle
tags:
  - kaggle
  - machine-learning
---

Coursera의 [How to Win a Data Science Competition](https://www.coursera.org/learn/competitive-data-science/home/welcome) 강의를 수강하고 있습니다. 혼자서 시행 착오를 하면서 하나씩 터득하는 시간을 아끼고 머신러닝 팁을 많이 배울 수 있는 강의라고 생각합니다. 이 글에서는 Week4의 Tips and Tricks 내용을 정리하겠습니다.

<br/>

## 1. Alexander Guschin

- 목표를 명확하게 해라.
  - 문제에 대해서 더 잘 알기 위해서
  - 소프트웨어 도구에 대해서 친숙해지기 위해서
  - 메달을 따기 위해서

<br/>

- Competition에 들어가서 Idea를 생각해봐라.
  - 몇몇 구조에서 아이디어를 조직화한다.
  - 가장 중요하고 유망한 아이디어를 선택한다.
  - 왜 무엇이 잘못되고 잘 되었는지 이해한다.

<br/>

- 모든 것이 Hyperparameter이다.
  - Importance
  - Feasibility
  - Understanding
    - 하나의 parameter를 변경하는 것이 전체 파이프라인에 영향을 준다.

<br/>

- Data Loading
  - 간단한 전처리를 하고 이를 hdf5 / npy 포맷으로 바꾼다. 훨씬 빠르다.
  - 64 비트 데이터로 원본 데이터가 저장되어 있으면 이를 32비트로 바꾸어서 저장한다.
  - 큰 규모의 데이터는 chunk로 처리할 수 있다.

<br/>

- Performance 평가
  - Extensive validation is not always needed.
  - 가장 빠른 모델부터 선택해라. (LightGBM) (Early Stopping을 이용해라.)
    - Baseline -> Current Solution -> Add and Change feature -> Optimize parameter -> new method 의 반복이다.
  - Feature Engineering을 다하고 나서 Stacking이나 모델 튜닝을 한다.

<br/>

- Fast and Dirty are. etter
  - 너무 많이 코드 질에 신경쓰지 마라.
  - 간단하게 유지하고 중요한 것만 저장한다.
  - 만약 주어진 computational resource에 불만족하다면 더 큰 서버를 빌려라.

<br/>

- Best practices from software development
  - 좋은 변수명을 사용한다.
  - 연구를 재생산가능하도록 유지한다.
  - 코드를 재사용한다.

<br/>

- Read papers
  - 머신러닝 페이퍼를 읽는다.
  - 새로운 feature를 생성하기 위해서 도메인에 관련된 페이퍼를 읽는다.

<br/>

- 전체 파이프라인
  - Read forum and examine kernel first.
  - Start with EDA and a baseline.
    - to make sure the data is loaded correctly
    - to check if validation is stable
  - I add new features in bulk
    - At start, I create all features I can make up
    - I evaluate many features at once
  - Hyperparameter optimization
    - 먼저 Train set에 대해서 Overfitting을 하고 나서 모델에 제약을 건다.

<br/>

- Code organization: 깨긋하게 정리한다.
  - Very important to have reproducible results.
  - Long execution history leads to mistakes.
  - 각 submission마다 새 notebook 및 git을 적용한다.
  - test와 validation을 csv로 젖아한다.
  - 전체로 학습하고 결과를 제출한다.
  - custom library

<br/>

## 2. KaxAnova

- Pipeline
  - 문제 이해(1일)
  - EDA(1-2일)
  - CV 전략을 정의한다.
  - Feature Engineering(마지막 3-4일 전까지)
  - Modeling(마지막 3-4일 전까지)
    - 혼자서 문제를 1주 정도 풀어보고서 다른 사람들의 kernel을 본다.
  - Ensemble(마지막 3-4일)
    - 모든 모델을 만들고 최고의 결과를 얻는다.

<br/>

- 문제 이해
  - Type of problem
  - How big is data?
  - Hardware needed(CPU / GPU / RAM)
  - Software needed
    - 각각의 competition마다 가상환경을 만든다.
  - 어떻게 metric이 테스트 되어야 할까.
  - 이전에 작성한 코드 중에 관련있는 것이 있는가.

<br/>

- EDA
  - 변수에 대한 히스토그램을 그린다.
    - 학습과 테스트 사이의 유사성을 본다.
  - feature와 target 사이의 그래프를 그린다.
  - 단변수 별 예측성 metric을 고려한다. (R square, auc)
  - 숫자 변수를 Binning하고 상관계수 행렬을 그린다.

<br/>

- CV 전략
  - 사람들은 올바르게 검증하는 방법을 선택했기 때문에 이긴다.
  - 시간이 중요한가?
    - Time based validation
  - 학습 데이터에 있는 entity별 차이가 있는가.
    - Stratified Validation
  - Completely random
    - Random validation

<br/>

- Feature engineering
  - 다른 문제는 다른 feature engineering 작업이 필요하다.
    - 다 알 수는 없으니 과거 문제를 보고 사람들이 어떻게 했는지 파악한다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-feature.png">
</p>

<br/>

- Model
  - 문제마다 다른 모델이 좋다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-modeling.png">
</p>

<br/>

- Ensemble
  - 내부 validation 결과와 테스트 결과는 저장되어야 한다.
  - 작은 데이터는 작은 ensemble 기술을 요한다.
  - 서로 연관되지 않은 prediction을 평균하는 것이 결과가 좋다.
  - 많은 데이터는 스태킹을 이용할 수있다.
  - 스태킹 절차는 모델링 절차와 유사하다.

<br/>

- Tips on collaboration
  - 더 재밌다.
  - 더 많이 배운다.
  - 더 좋은 성적을 낸다.
  - 적어도 2개의 방식으로 생각하게 된다.
  - dynamics를 이해하고 collaborating을 시작한다.
  - 내 rank 주변에 있는 사람과 협업한다.
  - 다른 방식으로 문제를 푼 사람과 협력한다.

<br/>

- Selection final submission
  - 보통, leaderboard와 local에서 가장 좋은 성능을 내는 모델을 선택한다.

<br/>

- Final Tip
  - 다른 사람들이 어떻게 했는지 본다.
  - 사람에게 질문한다.
  - Notebook을 함수화한다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-tip.png">
</p>



