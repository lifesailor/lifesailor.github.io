---
title: Deep Learning 정리(2) - Deep Learning Fundamental
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념이 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. 하지만 해당 글에서는 구현을 얘기하기 전에 간략하게 딥러닝의 기본을 짚고 넘어갈 생각입니다. 이 글을 작성하는 데 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의 및 Standford의 [CS231n](http://cs231n.stanford.edu/) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

앞으로 딥러닝을 정리하면서 크게 3가지 부분으로 나누어서 정리를 해보려고 합니다. 

1. 딥러닝이란 무엇이고 어떻게 동작 하는가.

2. 딥러닝을 동작하게 하려면 어떻게 해야 하는가.

3. 딥러닝이 잘 동작하게 하려면 어떻게 해야 하는가.

<br/>

## 1. 딥러닝이란 무엇이고 어떻게 동작 하는가.

딥러닝 공부를 할 때 먼저 딥러닝 구조를 배우게 됩니다. 딥러닝은 다음 그림과 같이 입력층, 은닉층, 출력층으로 구성되어 있습니다. 한 번에 이 구조를 모두 파악하기는 어렵기 때문에 단계적으로 하나씩 층을 늘려가면서 설명하겠습니다.

- 은닉층이 없는 딥러닝: Logstic Regression
- 은닉층이 1개 있는 딥러닝: Two Layer Network
- 은닉층이 여러 개 있는 딥러닝: Deep Neural Network

<br/>

## 2. 딥러닝이 동작하게 하려면 어떻게 해야 하는가.

층을 무조건 쌓는다고 해서 딥러닝이 잘 동작하는 것은 아닙니다. 딥러닝이 잘 동작하려면 층 사이의 가중치와 연산에 대해서 잘 정의 할 필요가 있습니다. 1980년대 이후에 딥러닝이 암흑기를 겪은 이유 중 하나가 당시에는 이 점에 대해서 충분히 고려하지 못했기 때문입니다. 

- 가중치 초기화: Weight Intialization
- 활성화 함수: Activation Function
- 최적화: Optimization

<br/>

## 3. 딥러닝이 잘 동작하게 하려면 어떻게 해야 하는가.

2에서 딥러닝을 잘 설정했음에도 불구하고 때로 딥러닝이 잘 동작하지 않을 때가 있습니다. 그럴 때 어떻게 해야 할까요? 3에서는 이 주제들을 다룹니다.

- 정규화: Regularization
- 하이퍼파라미터 최적화: Hyperparemter Tuning
- 기타 머신러닝 전략: Machine Learning Strategy

<br/>

위의 3가지 내용을 잘 숙지하고 나면 딥러닝 모델의 가장 기본적인 부분을 이해할 수 있고 해당 지식을 바탕으로 자신이 해결하고 싶은 문제에 특화해서 공부를 더 해나가시는 데 문제가 없을 것이라고 생각합니다.



