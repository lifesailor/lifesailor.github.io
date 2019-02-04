---
title: "머신러닝 개념 정리(1) - What is machine learning"
categories:
  - machine learning 
tags:
  - machine learing
  - concept
---

앞으로 머신러닝 모델을 하나씩 정리하고자 합니다. 하지만 그 전에 앞서 머신러닝에 대한 기본적인 개념을 몇 개의 글에 거쳐서 정리할 생각입니다. 처음으로 '머신러닝이란 무엇인가'에 대해서 정리해보겠습니다. 이 글을 작성하면서 Ian GoodFellow의 [deeplearningbook](http://www.deeplearningbook.org/)과 Aurélien Géron의 [핸즈온 머신러닝](https://book.naver.com/bookdb/book_detail.nhn?bid=13541863)을 참고했습니다.

<br/>

## 1. 머신러닝의 정의

머신러닝은 명시적으로 프로그램하지 않고 데이터로부터 학습하는 알고리즘입니다. 사람이 직접 규칙을 작성하는 알고리즘과 달리 머신러닝 알고리즘은 데이터로부터 직접 규칙을 찾아냅니다. 인공지능 전문가들은 머신러닝에 대해서 다음과 같이 정의를 내립니다.

- The field of study that gives computers the ability to learn without being explicitly programmed (Author Samuel)

- A computer program is said to learn from experience E with respect to some class of Tasks T and performance measure P, if its performance at tasks in T, as measured by P,  improves with experience E. (Tom Mitchell)

<br/>

## 2. 머신러닝 프로세스

항상 머신러닝을 사용해야 하는 것은 아닙니다. 규칙 기반 시스템이 적절한 경우도 있고 머신러닝 기반 시스템이 적절한 경우도 있습니다. 예를 들어서 회계 시스템과 같이 정해진 순서를 따라서 진행해야 하는 시스템은 머신러닝보다는 규칙에 의해서 프로그램되어야 합니다. 반면에 스팸 필터와 같이 규칙을 생각하기 복잡하거나 계속해서 규칙이 변해야 할 때는 머신러닝 시스템이 더 나은 선택일 수 있습니다. 2000년대의 욕설과 현재의 욕설은 다르기 때문에, 스팸 필터를 규칙 기반 시스템으로 만든다면 계속해서 규칙이 주렁주렁 달린 유지보수하기 어려운 시스템이 될 수 있습니다. 스팸 필터를 머신러닝 시스템으로 구성한다면 아래 그림과 같이 데이터를 더 넣어주는 것만으로 새로운 스팸을 계속해서 업데이트 할 수 있습니다.

![](/assets/images/machine-learning/machine-learning-2.png)

<br/>

## 3. 머신러닝과 통계의 관계

머신러닝을 공부하는 사람들은 머신러닝과 통계는 무슨 관계인지 한 번쯤 생각해보게 됩니다. 머신러닝과 통계 모두 데이터로 부터 실제 함수 $y=f(x)$에 대해서 추정(estimate)하고자 시도합니다. 하지만 머신러닝과 통계는 서로 강조하는 점이 다릅니다. 

- 머신러닝은 실제 $y = f(x)$를 추정함에 있어서 실제 $y$ 값과 가장 차이가 적은 함수 $f^{hat}$을 구하는 데 초점을 둡니다. 
- 통계는 $y = f(x)$ 를 추정한 함수 $f^{hat}$ 이 정말 유의한 지 검정하는 데 초점을 둡니다. 

<br/>

## 4. 머신러닝이 다루는 세계 - Uncertainty

머신러닝이 데이터로부터 실제 함수 $y=f(x)$를 추정할 때 반드시 불확실성을 수반합니다. 즉, $y=f(x)$ 함수 자체를 정확하게 추정할 수는 없습니다. 불확실성은 다음의 이유로 존재합니다.

- 불확실성은 다음과 같은 이유로 존재한다.
  - 우리가 추정하려는 $y = f(x)$에서 모든 $(x, y)$ 쌍을 보고 추정하는 것이 아니기 때문입니다. 
  - $y = f(x)$에서 $f$ 라는 모델을 사용하는 가정 자체에서 정보 손실이 있습니다.

따라서 $y = f(x)$ 를 추정함에 있어서 y = $f^{hat}(x) + \epsilon$, 항상 $\epsilon​$ 이 존재할 수 밖에 없습니다. 이를 irreducible error라고 표현합니다.

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/machine-learning/machine-learning-1.png">
</p>
<br/>

## 5. 머신러닝 기반 이론

머신러닝 다음 이론들을 기반으로 정립됩니다.

- Probability Theory - 불확실성을 수량화해서 표현합니다.
- Decision Theory - 확률 이론을 바탕으로 최적의 모델을 선택하는 기준을 제시합니다. 
- Information Theory - 실제 $y = f(x)$와 내가 추정해서 얻은 $y = f^{hat}$과 오차에 대한 척도를 제공합니다.
  - Decision Theory에서 사용하는 Loss Function의 근간이 됩니다.

<br/>

위의 3번에서 머신러닝과 통계는 $y = f(x)$를 추정(estimate)하는 것이라고 했습니다. 도대체 추정(estimate)란 무엇이고 추정량(estimator)이란 무엇일까요?

