---
title: 딥러닝 정리(4) - Regularization
categories:
  - deep learning
tags:
  - machine learning
  - deep learning

---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

Regularization은 머신러닝 모델이 training set에 과적합 되지 않고 모델이 보지 않은 데이터 셋에 좋은 성능을 내도록 하는 것입니다. 즉, 정규화는 모델의 일반적인 성능을 높이려는 시도입니다. 많은 정규화 방법들이 있습니다. 다음은 Deeplearning book 7장에 나오는 정규화 방법을 나열한 것입니다.

<br/>

1. Paramter Norm Penalties
2. Dataset Augmentation
3. Noise Robustness
4. Semi-supervised Learning
5. Multitask Learning
6. Early Stopping
7. Paramter Typing and Paramter Sharing
8. Sparse Representation
9. Bagging and Ensemble Method
10. Dropout
11. Adverserial Training

<br/>

이제부터 위의 정규화 방법에 대한 간단한 소개를 하나씩 작성해보겠습니다.

<br/>

1. Paremeter Norm Penalties

<br/>

![](/assets/images/deep-learning/regularization/weight-decay.png)

<br/>

왼쪽 그림의 빨간색 선을 보면 parameter가 너무 커짐으로써 training 데이터에 overfitting된 것을 볼 수 있습니다. Parameter Norm Penalty는 Cost뿐만이 아니라 Parameter Norm을 같이 줄이도록 시도함으로써 paramter가 너무 커지는 것을 방지합니다. Norm의 경우에는 $\mid w\mid^1 , \mid w \mid ^2$ 둘 다 사용할 수 있는데 여기에서는 $\mid w \mid^2$을 가정하고 코드를 작성하였습니다. 

$$J(w) = J(w) + \lambda*w^tw$$

$$w = w - \alpha(\lambda w + dJ)​$$

```python
"""
cost
"""

# 기존 cost
cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

# 변경 cost - 여기에서 W는 각 parameter를 의미합니다.
cost =  - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) - lambda_ * (1 / 2m) * np.sum(np.multiply(W, W))
```

```python
"""
update parameter
"""

# 기존 parameter 업데이트
W = W - alpha * dW

# 변경 parameter update 
W = W - alpha * (lambda_ * W + dW) 
```

<br/>

2. Dataset Augmentation

