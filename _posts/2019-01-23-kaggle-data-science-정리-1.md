---
title: How to Win a Data Science Competition - Week 1 정리
categories:
  - kaggle
tags:
  - kaggle
  - machine-learning
  - coursera
---

Coursera의 [How to Win a Data Science Competition](https://www.coursera.org/learn/competitive-data-science/home/welcome) 강의를 수강하고 있습니다. 혼자서 시행 착오를 하면서 하나씩 터득하는 시간을 아끼고 머신러닝 팁을 많이 배울 수 있는 강의라고 생각합니다. 이 글에서는 Week1 내용을 정리하겠습니다.



## 1. Overview

강의 전체 내용을 간략하게 소개합니다.



-  Week1
  - Introduction to competition
  - Feature preprocesssing & Extraction
- Week2 
  - EDA
  - Validation
  - Data leaks
- Week3
  - Metrics
  - mean-encoding
- Week 4
  - Advanced features
  - Hyperparameter optimization
  - Ensembling
- Week5
  - Final Project

<br/>

## 2. Data Science Competition

실제 환경에서 문제를 푸는 것과 Competition에 참여하는 것은 큰 차이가 있습니다. 그럼에도 불구하고 Competition은 데이터 사이언스에 대한 여러 가지를 배울 수 있는 기회입니다. Competition에 참여하는 것은 단지 알고리즘 자체를 튜닝하는 것만이 아니고, 통찰력이 있어야 하고 창의적으로 문제에 접근해야 합니다.

<br/>

- Competition 하는 이유
  - 배우고 네트워킹하기에 좋은 기회다.
  - 최신의 기법을 배울 수 있다.
  - 데이터 사이언스 커뮤니티에서 유명해질 수 있다.

<br/>

- 실제 환경과 Competition에서 문제 풀이
  - 실제 환경
    - Understand business problem
    - Problem formalization
    - Data collecting
    - Data preprocessing
    - Modeling
    - Way to evaluate model in real life
    - Way to deploy model
  - Competition
    - Data preprocessing
    - Modeling

<br/>

- 실제 환경과 Competition에서 주의해야 할 점
  - 실제 환경
    - Problem formalization
    - Choice of target metric
    - Deployment issue
    - Inference speed
    - Data collecting
    - Model complexity
    - Target metric value
  - Competition
    - (Model complexity) Target metric value 정도이다.

<br/>

## 3. Feature preprocessing and generation

Feature를 전처리하는 방법과 생성하는 것은 중요합니다. 각각의 방법에 대한 효과는 모델마다 다릅니다. 

<br/>

### 1. Numeric Features

- 모델에 따른 차이
  - Tree-based method는 보통 영향을 받지 않는다.
  - Non tree-based method는 영향을 받는다.



- Feature Preprocessing 방법
  - Scaling: min max scaling, standard scaling
  - Outlier: 특정 percentile 밖의 outlier 제거(histogram)
  - Rank: Outlier가 있을 시에 minmax scaler 쓰는 것보다 낫다.
  - Transformation: log transform, power transform
    - neural network 등에 대해서 성능을 높일 수 있다.



- Feature Generation 방법
  - Data에 대한 이해와 창의성이 Feature Generation의 핵심이다.
  - 예시
    - 부동산 가격, 평수로부터 '평당 가격' 변수 생성
    - 수원지까지 x, y좌표로 부터 '거리' 변수 생성

<br/>

### 2. Categorical and Ordinal Features

- Feature 종류
  - Ordinal Features
    - Ticket: p1, p2, p3
    - Driver's license: A, B, C, D
  - Categorical Features
    - Sex



- Feature Preprocessing 방법
  - Label Encoding: 숫자로 인코딩(Tree-based model에서만 사용)
  - Frequency Encoding: 비율로 인코딩(Tree-based, Non tree-based model에 사용)
    - Frequency 자체가 Target Value에 상관관계가 있을 수 있다.
    - 예시: [S, O, Q] = [0.5, 0.3, 0.2]
  - One-hot encoding: Categorical features



- Feature Generation 방법
  - Feature Interaction을 만든다.
    - 예를 들어서, 성별(2 class)과 등급(3 class)이 있으면 이를 조합해서 6class 변수를 만든다.

<br/>

### 3. Datetime and Coordinates

- Datetime(시간)
  - Periodicity
    - Day number in week, month, season, year, second, minute, hour
  - Time Since
    - row independent moment
      - 예를 들어, 1970년 1월 1일부터 현재까지 시간
    - row dependent moment
      - 휴일로부터 얼마 남았는가.
      - 휴일로부터 얼마 지났는가.
  - Difference between dates
    - 마지막 방문일과 이전 방문일 사이의 날짜 수



- Coordinates(공간 좌표)
  - Other train samples and center of center of cluester
  - Aggregated stats
  - Interesting places from train / test data on additional data

<br/>

### 4. Missing Value

- Hidden NaN
  - Nan이 아니어도 -1, -999 등으로 되어 있을 수 있다. 이를 조심해야 한다.



- Filling NaN
  - -999, -1: 의미 없는 값으로 둔다. (linear model의 경우 영향 받을 수 있다)
  - mean, median: 
  - Reconstruct value



- Feature generation with missing value
  - missing value 있을 때 feature generation 하는 경우는 극도로 조심해야 한다.

<br/>

![](/assets/images/kaggle/top-kaggler/1.png)











