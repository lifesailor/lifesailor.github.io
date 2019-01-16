---
title: "Data Science Interview 정리 - 분석 일반(1)"
categories:
  - data science interview
tags:
  - data science
  - interview
---

## 1. 좋은 feature란 무엇인가요? 이 feature의 성능을 판단하기 위해서는 어떤 방법이 있나요?

통계학과 머신러닝에서는 2가지 변수가 있습니다. 우리가 궁극적으로 알고 싶은 목적에 해당하는 Y 변수와 이를 설명하는 X 변수가 있습니다. Feature는 두 가지 중 X 변수를 의미합니다. 

좋은 Feature란 목적인 Y를 잘 설명할 수 있는 X 변수입니다. 예를 들어서 서울의 집 값(Y)을 예측하는 문제에서 1) 집의 위치나 2) 집의 크기는 집 값을 잘 설명할 수 있는 좋은 feature이지만 3) 집을 사는 사람의 발 사이즈는 일반적으로 집 값과 무관한 좋지 않은 변수라고 할 수 있습니다.

데이터 사이언티스트가 모델에 데이터를 넣기 전에 해야 하는 일 의 하나가 의미 있는 Feature를 만들거나 선택하고 전혀 의미가 없는 Feature는 제거함으로써 모델이 X, Y의 관계를 잘 찾도록 도와주는 것입니다.



Feature의 성능을 정량적으로 판단하는 가장 기본적인 방법으로는 일변량 통계를 활용하는 방법이 있습니다. Feature 하나하나를 독립적으로 바라 보고 각각의 Feature가 Y를 얼마나 잘 설명하는가를 판단합니다. 판단의 기준은 ANOVA F value입니다. sklearn 라이브러리에는 해당 기능이 구현되어 있습니다.

```python
# 라이브러리 불러오기
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split

# 데이터 읽기
cancer = load_breast_cancer()

# Train, Test 분리하기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0, test_size=.2
)

# ANOVA F-values를 바탕으로 상위 50%의 변수만 선택한다.
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print("X_train shape: {}".format(X_train.shape))
print("X_train selected shape: {}".format(X_train_selected.shape))

# 결과
X_train shape: (455, 30)
X_train selected shape: (455, 15)
```

위의 방법은 처음 시도해볼 수 있는 가장 간단한 방법이지만 Feature간의 관계를 독립적으로 상정한다는 점에서 한계가 있습니다. 

다른 방법으로는 머신러닝 모델을 사용해서 Feature 성능을 평가하는 방법이 있습니다. Tree 모델은 모델 자체에서 특성 중요도를 산출할 수 있습니다. 즉 모든 Feature를 Input으로 넣고 모델에 학습함으로써 어떤 Feature를 중심으로 가장 많이 분기되었는지 확인할 수 있습니다.

![](/Users/lifesailor/Desktop/project/lifesailor/github/data-science-interview-answer/image/feature-importance.png)

모델의 설명력과 복잡도 간의 관계를 바탕으로 Feature의 유용성을 판단할 수도 있습니다. 회귀 분석에서는 Forward Selection, Backward Selection, Stepwise Selection등의 방법을 사용해서 Feature를 선택합니다. 해당 방법들은 모델의 설명력이 증가하는 정도와 모델의 복잡성이 커지는 정도를 AIC, BIC와 같은 기준으로 비교해서 해당 Feature의 유용성을 판단합니다.



이 외에도 각 Feature의 분포를 시각화를 하고 Feature간의 상관관계를 보는 것도 가장 많이 쓰는 방법 중의 하나입니다. 각 Feature 또는 Feature와 Target 간의 관계를 그려보면 하나의 수치로 발견하지 못했던 유의미한 정보를 찾아낼 수 있습니다. 상관관계는 각 Feature간의 관계를 하나의 수치로 파악할 수 있어서 처음에 대략적인 감을 잡는데 용이합니다.



Deep Learning은 Feature를 찾는 과정까지 학습하는 방법입니다. 그럼에도 불구하고 사람이 Y를 잘 설명할 수 있는 Feature를 강조해준다면 Deep Learning 역시 훨씬 더 빠르게 학습하고 성능 또한 높일 수 있습니다. 특히 데이터가 충분하지 않은 경우에는 좋은 Feature를 설계할 수 있는 고민이 필요합니다.



참고: 파이썬 라이브러리를 활용한 머신러닝 | 안드레아스 뮐, 세라 가이도 | 한빛미디어, 2017

<br/><br/>



## 2. “상관관계는 인과관계를 의미하지 않는다”라는 말이 있습니다. 설명해주실 수 있나요?

상관관계는 두 변수의 선형적 관계를 나타내는 지표이고 인과관계는 두 변수간의 관계를 **원인과 결과로 해석**하는 것입니다. 



우선, 상관 관계는 두 변수간의 선형적 관계를 의미합니다. 즉, X 변수가 증가할 때 Y 변수 또한 증가한다면 양의 상관관계를 가진다고 말합니다. 반대로 X 변수가 증가할 때 Y 변수가 감소한다면 음의 상관관계를 가진다고 말합니다.



예를 들어서 아이스크림이 팔리는 개수가 많아지면서 집에서 지출하는 전기세가 올라간다면, X: 아이스크림 팔리는 개수, Y: 집의 전기세 두 가지 변수는 양의 상관관계가 있다고 할 수 있습니다. 하지만 아이스크림이 많이 팔리기 때문에 전기세가 올라가는 것도 아니고 전기세가 올라가기 때문에 아이스크림이 많이 팔리는 것도 아닙니다. 즉, 두 변수간의 상관관계가 있다고 해서 이를 **원인과 과 결과 관계로 해석**해서는 안 됩니다.

<br/><br/>



## 3. A/B 테스트의 장점과 단점, 그리고 단점의 경우 이를 해결하기 위한 방안에는 어떤 것이 있나요?

