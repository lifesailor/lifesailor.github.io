---
title: Deep Learning 정리(6) - Improving Performance Strategy
categories:
  - deep learning
tags:
  - machine learning
  - deep learning

---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념이 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. 이 글을 작성하는 데 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의 및 Standford의 [CS231n](http://cs231n.stanford.edu/) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

지난 글에서는 딥러닝의 성능을 높이기 위한 구체적인 기술을 요약했습니다. 하지만 지난 글에서 언급한 기술을 사용한다고 해서 모델의 성능이 항상 최적화되는 것은 아닙니다. 모델의 성능을 높이기 위해서는 여러 번 실험을 반복해야 합니다. 이번 글에서는 딥러닝 모델 실험을 하기 위한 방법을 정리해보았습니다.

<br/>

## 1. Orthogonalization

머신러닝 실험할 때는 Orthogonal Control을 해야 한다. 즉 나머지 요인들을 고정해두고 하나씩 변경해야지 해당 파라미터의 효과에 대해서 알 수 있다. 그렇지 않으면 무엇 때문에 성능이 좋아지거나 나빠졌는지 이해하지 못할 수 있다.

<br/>

Machine Learning은 다음 작업 순서를 거친다. 먼저, Training set에 잘 작동하게 만든다. 이후에 Dev set의 성능을 높이고 나서 Test set에서 높은 성능을 내는지 검증한다. 만약 그렇다면 실제 환경에서 잘 동작하는지 모니터링 한다.

<br/>

1가지 평가 지표를 사용하는 것이 좋다. 만약에 여러가지 조건을 만족해야 하면 1가지 평가 지표과 몇 개의 만족 조건을 사용하는 것이 대안이 될 수 있다. 예를 들어서, 속도는 10ms 이하라는 만족 조건 하에서 정확도를 평가 지표로 하는 것을 생각해볼 수 있다.

<br/>

## 2. Improving model performance

지도 학습은 다음 2가지 순서를 따른다. 

- Training set에 fit 한다. (avoidable bias)

- Dev / Test에 generalize 한다. (variance)

<br/>

Bias, Variance를 판단함으로써 어떻게 성능을 최적화 해나가야 하는지 선택할 수 있다.
- Human level: 4%
- Training error: 7%(Avioidable bias)
  - Train bigger model
  - Train longer / better optimization algorithm
  - NN architecture / hyperparameter search
- Dev error(Variance)
  - More data
  - Regularization
  - NN architecture 

<br/>

## 3. Hyperparameter Tuning

Hyperparameter Tuning은 다음과 같은 순서를 따라서 진행된다.
- 1순위: Learning rate
- 2순위: Hidden node 개수 및 mini-batch 크기
- 3순위: Layer 개수 및 learning rate decay

<br/>

Grid 탐색보다는 Random 탐색을 하는 것이 좋다.
- Coarse하게 샘플로 실험하고 영역을 찾고 나서 추가적으로 실험한다.

<br/>

Hyperparameter를 선정할 때 적절한 크기를 사용해서 실험한다. 
- learning rate 등을 설정할 때 그냥 uniform 분포 대신에 $\alpha = 10^{r}, r \in [-4, 0]$을 사용한다.
  - 우리가 더 궁금한 것은 0.0001 & 0.0002의 차이가 아니다.

<br/>

## 4. Training, Dev, Test set

Dev set 및 Test set은 같은 분포에서 나온 데이터로 만들어야 한다. Dev set 및 Test set에 잘 동작하지만, 실제 환경에서 잘 동작하지 않으면, Dev set 및 Test set를 바꿔야 한다.
- 가장 중요한 것은 실제 환경에서 잘 동작하는 것이다. 미래에 실제 환경에 적용할 데이터를 기준으로 Dev set, Test set을 선택해야 한다.

<br/>

과거와 달리, 1,000,000개의 데이터가 있으면 Training set / Dev set / Test set 비율을 6:2:2 대신에 980,000, 10,000, 10,000개의 구성을 한다.
- Test 사이즈는 시스템의 신뢰도를 측정할 수 있을 만큼 커야 한다.

<br/>

Training set과 Dev set 및 Test set이 성능이 다르면 2가지 이유가 있을 수 있다.
- High Variance
- Different distribution
  - 이를 확인하기 위해서 Training 데이터의 일부를 떼어서 Training-dev set으로 실험할 수 있다.

<br/>

전체 요약
- Human error: 4%
- Training set error: 7% (Avoidable bias)
- Training dev set error: 10% (Variance)
- Dev set error: 12% (Data mismatch)
- Test error: 15% (degree of overfitting to dev set)

<br/>

## 5. Transfer Learning & Multitask Learning

Transfer Learning은 데이터가 충분하지 않을 때 다른 모델을 기반으로 현재 데이터에 맞게 튜닝하는 방법이다. Transfer Leanring은 다음 경우에 사용한다.
- A, B에서 X(입력)이 동일하다.
- A가 B보다 데이터가 많다.
- A에서 배운 low level feature가 B를 배우는 데 도움이 된다.

<br/>

Multi-task learninig은 각각의 task들을 동시에 학습시켜 예측 성능을 향상시키는 방법이다. Multi-task learning은 task의 수가 많고 각 task에 속한 샘플의 수가 적을 때 일반적으로 task마다 모델을 생성하는 single-task learning보다 높은 성능을 보여준다고 알려져 있다. Multi-task Learning은 다음 경우에 사용한다.

- 공유된 Low level feature를 가지는 것이 도움이 된다.

- 각 Task에 대한 데이터가 거의 비슷하다.
- 큰 규모의 신경망을 운용할 때 작동한다.

<br/>

## 6. End-to-End deep learning

딥러닝을 사용하기에 앞서서 다음 질문을 해야 한다. **End-to-End deep learning을 하기 위한 충분한 데이터가 있는가.** 그렇다면 딥러닝을 시도해볼만 하고 아니면 단계적으로 나누어서 학습하는 것이 좋다. 문제 상황에 맞게 전략적으로 결정해야 한다.

상황에 맞게 전략적으로 결정해야 한다.

- End-to-End로 학습할 것인지 결정해야 한다.
- 단계를 나누어서 학습할 것인가.



End-to-End deep learning의 장단점

- 장점
  - Feature를 직접 만들지 않아도 된다.

- 단점
  - 데이터 양을 많이 필요로 한다. 
  - 수작업으로 만들 수 있는 유용한 컴포넌트를 사용하지 못한다. 

<br/>

## 7. 전체 프로세스

데이터를 훈련, 검증, 테스트 세트로 나누고 하나의 평가 지표를 선정한다. 그리고 나서 첫 시스템을 빠르게 개발한다. (Build Fast and iteratate) 이후 부터는 Bias. Variance 및 Error Analysis를 통해서 성능을 개선시켜 나간다.