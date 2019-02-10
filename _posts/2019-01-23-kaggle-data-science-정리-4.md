---
title: How to Win a Data Science Competition - Week 4 - Ensemble
categories:
  - kaggle
tags:
  - kaggle
  - machine-learning

---

Coursera의 [How to Win a Data Science Competition](https://www.coursera.org/learn/competitive-data-science/home/welcome) 강의를 수강하고 있습니다. 혼자서 시행 착오를 하면서 하나씩 터득하는 시간을 아끼고 머신러닝 팁을 많이 배울 수 있는 강의라고 생각합니다. 이 글에서는 Week4의 Ensemble 내용을 정리하겠습니다.

<br/>

## 1. 간단한 Ensemble 종류

- Averaging(or blending)
  - $\frac{1}{2}(model1 + model2)$
- Weighted Averaging
  - $0.7 model1 + 0.3 model2$
- Conditional Averaging
  - Age가 50보다 작을 때 model1이 좋고 Age가 50보다 클 때 model2가 좋으면 conditional Averaging을 한다.

<br/>

## 2. Bagging

- 배깅은 같은 모델의 약간 다른 버전을 평균 내어 정확도를 높이기 위해서 사용한다.
- Error는 Bias와 Variance로 나눌 수 있다.
  - Bagging은 Variance를 줄이는 방법이다.
- Bagging을 통제하는 Parameter
  - seed를 변경한다.
  - row subsampling or bootstraping
  - shuffling
  - column subsampling
  - model-specific parameters
  - number of models
  - parallelism

<br/>

## 3. Boosting

- 부스팅은 각각의 모델이 이전 모델의 퍼포먼스를 고려하면서 순차적으로 평균내어지는 것이다.
- 2개의 main type이 있다.
  - weight based
  - residual based 

- Weighted based boosting
  - 각각의 훈련 Sample에 Weight를 준다. 못 맞춘 Sample에 높은 Weight를 준다.
  - 중요 parameter
    - learning rate
    - number of estimators
    - input model(이론상 weight를 줄 수 있는 모델이면 상관 없다. 하지만 tree를 주로 사용한다.)
    - Adaboost, Logitboost
- Residual based boosting
  - 잔차에 fitting 한다.
  - 중요 parameter
    - learning rate
    - number of estimators
    - Rowsampling, column sampling
    - input model - better to be tress
    - Sub boosting type
      - Gradient Bassed Learning
    - Sklearn Gradient Boosting Machine, XGBoost, LightGBM, H20 GBM, Catboost

<br/>

## 4. Stacking

- Stacking은 많은 수의 모델을 사용해서 예측을 하고 다양한 meta model을 사용해서 prediction에 훈련한다.
- 기본적인 방법
  - Split train set into 2 disjoint sets
  - Train several base learners on the first part
  - Make prediction with base learners on the second part
  - Use predictions from the above as the input to train a higher level learner

- 생각해야 할 점
  - time sensitive data인가
  - 다양성이 성능에 중요하다.
    - 다양성은 다음 요인에서 온다: different algorithm, different input features, meta model is normally modest
  - N model 이후에 성능이 수렴한다.

<br/>

## 5. StackNet

- A scalable meta modeling methodology that utilizes stacking to combine multiple models in a neural network architecture of multiple levels

- 왜 사용하는가.
  - Competition에서 이기기 위해서 사용한다. Deep Stacking이 유용한다.
- StackNet은 NN처럼 사용한다.

- 학습 방법
  - KFold를 사용한다.
  - Tree 모델은 저체 데이터를 사용하는 것이 더 잘 맞고 NN 모델은 KFold개의 모델을 Averaging하는 것이 더 잘 맞는다. (경험적)

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-stacking1.png">
</p>

<br/>

## 6. Stacking Tip

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-stacking2.png">
</p>

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/kaggle/anova-stacking3.png">
</p>

<br/>

## 7. Stacking 구성 방법

- [다음을 참조하세요](https://www.coursera.org/learn/competitive-data-science/supplement/JThpg/validation-schemes-for-2-nd-level-models)