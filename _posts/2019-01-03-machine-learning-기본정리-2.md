---
title: "머신러닝 개념 정리(2) - estimator"
categories:
  - machine learning 
tags:
  - machine learing
  - concept
---

앞으로 머신러닝 모델을 하나씩 정리하고자 합니다. 하지만 그 전에 앞서 머신러닝에 대한 기본적인 개념을 몇 개의 글에 거쳐서 정리할 생각입니다. 지난 글에 이어서 '추정량이란 무엇인가'에 대해서 정리해보겠습니다. 



## 1. Estimator

앞의 글에서 머신러닝과 통계는 $y=f(x)​$를 만족하는 함수 $f​$를 알아내는 것이라고 했습니다. 머신러닝과 통계에서는 주로 $x​$ 로 부터 $y​$ 를 설명할 때 특정 모수 $\theta​$ 와 함수 기본 형태를 가정하고 $y = f(x)​$를 구하고자 합니다.  $y=f(x)​$를 $x​$, $y​$ 만 가지고서 함수 관계를 추정하는 것은 어렵지만 숨어 있는 모수 $\theta​$가 있다고 가정하면 모델을 세워 수학적으로 접근해서 $y = f(x; \theta)​$로 문제를 풀 수 있기 때문입니다. 이러한 가정 하에서 함수 $f​$를 직접적으로 추정하는 문제는 $\theta​$ 를 추정하는 것으로 바뀝니다. 이 때, $\theta​$ 를 근사적으로 구하는 것이 바로 추정량(Estimator) $\hat{\theta}​$ 입니다.

<br/>

## 2. Good Estimator

그렇다면 좋은 추정량 $\hat{\theta}$ 가 되기 위한 조건은 무엇이 있을까요? [위키피디아](https://ko.wikipedia.org/wiki/%EC%B6%94%EC%A0%95%EB%9F%89)에 이에 대한 설명이 잘 나타있습니다. 좋은 추정량의 대표적인 성질은 Unbiasedness, Consistency, Efficiency 등이 있습니다. 각각의 성질에 대해서 간단히 알아보겠습니다.

- Unbiasedness(무편향)
  - $Bias(\hat{\theta})$는 $E(\hat{\theta}) - \theta$ 로 정의됩니다. Unbiasedness는 말 그대로 Bias가 없는, 즉 $Bias(\hat{\theta}) = 0$ 인 것을 의미합니다. 우리가 표준편차 추정량으로 표본 분산을 구할 때  n이 아니라 n-1로 나누는 이유는 이 Unbiaseness 조건 때문입니다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/machine-learning/biasedness.png">
</p>

- Consistency(일치성)
  - Consistency는 표본의 수가 커질 수록 추정량이 모수에 수렴한다는 것입니다. 이를 수학적으로 표현하면 다음곽 같이 할 수 있습니다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/machine-learning/consistency.png">
</p>

- Efficiency(효율성)
  - 두 추정량 $\hat{\theta}_1, \hat{\theta}_2$ 이 같은 모수 $\theta$ 에 대한 추정량이라고 할 때, 둘 중 표본편차가 작은 추정량이 더 효율적인 추정량입니다.

<br/>

## 3. 추정량을 구하는 방법

이제 추정량이 무엇인지 알게 되었고 어떤 성질을 가진 추정량이 좋은 추정량인지도 알게 되었습니다. 그렇다면 어떻게 좋은 추정량을 구할 수 있을까요? 가장 많이 사용되는 방법이 바로 머신러닝에서 자주 나오는 Maximum Likelihood Estimation 방법입니다. 



Maximum Likelihood Estimation는 크게 2가지를 가정합니다. 먼저, 우리가 가진 데이터가 어떤 분포에서 나왔다고 가정합니다. 또한, 각각의 데이터가 해당 분포에서 독립적으로 나왔다고 가정합니다. 위 가정 하에서 우리가 가진 데이터가 나올 수 있는 가장 높은 확률을 가진 추정량이 바로 Maximum Likelihood Estimator입니다. 이를 수식으로 표현하면 다음과 같습니다.

$$ \hat{\theta}_{MLE} = argmax_{\theta}P(X \mid \theta) = argmax_{\theta} P(X1\mid \theta)*P(X2\mid \theta) \dots P(Xn \mid \theta) ​$$



Maximum Likelihood Estimation은 Consistency, Efficiency를 보장하는 추정 방법이지만 Unbiasedness는 만족하지 않아서 보정을 해주게 됩니다. Maximum Likelihood Estimation 외에도 Least Square, Moment 기법과 같이 추정량을 구하는 다른 방법도 있습니다.

<br/>

위에서 제가 추정량을 설명하면서 마치 $\hat{\theta}$ 이 하나의 값인 것처럼 설명했습니다. 하지만 꼭 $\hat{\theta}$ 가 하나의 값이어야 할까요? 베이지안 주의자들은 이 질문에 대해서 'No!' 라고 답변합니다.

