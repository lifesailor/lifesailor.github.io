---
title: How to Win a Data Science Competition - Week 4 - Hyperparameter Tuning
categories:
  - kaggle
tags:
  - kaggle
  - machine-learning
---

Coursera의 [How to Win a Data Science Competition](https://www.coursera.org/learn/competitive-data-science/home/welcome) 강의를 수강하고 있습니다. 혼자서 시행 착오를 하면서 하나씩 터득하는 시간을 아끼고 머신러닝 팁을 많이 배울 수 있는 강의라고 생각합니다. 이 글에서는 Week4의 Hyperparameter Tuning 내용을 정리하겠습니다.



### 1. Hyperaparemeter Tuning 순서

- 가장 중요한 파라미터를 선택한다.
- 어떻게 해당 파라미터가 학습에 영향을 주는지 이해한다.
- 파라미터를 튜닝한다.
  - manually
  - automatically

<br/>

### 2. Hyperparameter Tuning - Tree Based models

- Xgboost 및 LightGBM

  - 숫자가 클 때 Overfitting 하는 요소
    - max depth(num_leaves)
    - subsample(bagging fraction)
    - colsample by tree 및 colsample by level (feature fraction)
    - eta(learning rate)
    - num_round(num iteration)
  - 숫자가 작을 때 Overfitting 하는 요소
    - min child weight(min_data_in_leaf)
    - lambda(lamba_l1, lambda_l2)

- RandomForest

  - Num estimator는 클 수록 좋다.
  - 숫자가 클 때 Overfitting 하는 요소
    - max depth
    - max features
  - 숫자가 작을 때 Overfitting 하는 요소
    - min samples leaf

  

단, 위에 있는 Parameter일수록 더 중요한 파라미터이다.

<br/>

### 3. Hyperparameter Tuning - Neural Network

- Framework

  - PyTorch
  - Keras

- Neural Network

  - 숫자가 클 때 Overfitting 하는 요소
    - Number of neurons per layer
    - Number of layers
    - Optimizer: Adam / Adadeleta(경험적으로 Overfitting)
    - Batch Size(Batch를 키우면 Learning Rate도 키우면서 실험한다.)
  - 숫자가 작을 때 Overfitting 하는 요소
    - Regularization
      - L2 / L1
      - Dropout
      - Static drop connect

  - Learning rate는 큰 것부터 작은 것으로 나아가는 게 좋다.

<br/>

### 4. Hyperparameter Tuning Tip

- 하이퍼파라미터 튜닝에 너무 많은 시간을 쓰지 않는다.
- 인내심을 가져라.
- Average everything
  - seed
  - over deviations from optimal parameter
    - 예를 들어서 max depth가 5가 최적일 때, 4, 5, 6을 취해서 평균을 취한다.

<br/>



