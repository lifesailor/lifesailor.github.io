---
title: Deep Learning 정리(1) - Machine Learning vs Deep Learning vs Human Brain
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

글 작성에 앞서 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의 및 Standford의 [CS231n](http://cs231n.stanford.edu/) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했다는 것을 밝힙니다. 

<br/>

## 1. Machine Learning

딥러닝이 기존 머신러닝과 가장 구별되는 지점은 feature를 선택하는 방식입니다. 기존의 머신러닝은 사람들 feature를 설계하는 경우가 많았습니다. 때로는 사람이 디자인 한 feature가 잘 작동했지만 그렇지 않은 경우도 있었습니다. 예를 들어서 어떻게 이미지 속의 '차'를 인식하기 위한 feature를 뽑을 수 있을까요? 사람은 배경, 색, 크기 등이 달라져도 이미지를 보자마자 '차'라는 것을 압니다. 하지만 컴퓨터를 활용해서 이미지 픽셀에서 차를 나타내는 feature를 뽑는 것은 쉬운 일이 아닙니다. 만약에 앞의 동그란 이 2개 있는 것이라고 차의 feature를 정의한다면 컴퓨터는 앞 부분이 가려진 차를 인식할 수 없을 것입니다. 이처럼 같은 차라고 해도 사람이 직접 배경, 색, 크기 등의 변동 요인(factor of variation)과 구분 지어서 feature를 디자인하는 것은 쉽지 않은 일입니다.

<br/>

## 2. Deep Learning

반면에, 딥러닝은 사람이 직접 feature를 설계하지 않습니다. 대신 딥러닝 모델이 feature를 뽑는 방법까지 학습합니다. 이를 표현 학습(representation learning)이라 합니다. 처음에는 딥러닝 모델이 선택한 feature이 잘못되었을 수도 있습니다. 그렇다면 딥러닝 모델도 '차'를 인식하지 못하겠죠. 하지만 딥러닝 모델은 계속해서 '차'를 제대로 인식할 수 있을 때까지 내부의 특징을 방법을 계속해서 수정합니다. 이 수정 과정에서 딥러닝 모델은 간단한 표현을 조합해서 점점 더 복잡한 표현을 표현하는 방식으로 학습한다고 알려져 있습니다. '차'의 특징을 알아내기 위해서 먼저 바퀴의 동그라미, 타이어의 줄무늬와 같은 간단한 표현을 조합해서 바퀴를 표현하고 이를 조합해서 '차'를 표현하는 방식으로 학습한다는 것이지요. 아래 그림은 다음 과정을 나타냅니다. (딥러닝 북에서 추출)

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/deep-learning/concept/deep.jpg">
</p>

<br/>

## 3. Human Brain

딥러닝 모델이 간단한 표현을 조합해서 점점 더 복잡한 표현을 표현하는 방식은 사람과 유사합니다. 1962년에 Hubel 과 Wissel은 고양이의 뇌가 '가장자리' 같은 간단한 구조에 반응하는 것을 발견했습니다. 즉, 고양이나 사람도 딥러닝처럼 간단한 구조를 먼저 인식하고 그것들을 조합해서 사물을 인식한다고 알려져 있습니다. 

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/deep-learning/concept/cat.jpg">
</p>

그렇다고 해서 딥러닝 모델이 사람이 뇌와 동일하게 작동하거나 학습한다고 생각하는 것은 위험합니다. 딥러닝은 수학 연산을 통해서 정보를 전달하지만 사람의 뇌는 화학 전기적인 상호작용을 통해서 정보를 전달합니다. 아직 사람이 뇌가 어떤 방식으로 학습하는지 정확하게 알려진 바가 없는 만큼, 인공 신경망이라는 딥러닝의 본명만큼 '인공'이라는 점을 기억하고 공부해나가면 좋을 것 같습니다.