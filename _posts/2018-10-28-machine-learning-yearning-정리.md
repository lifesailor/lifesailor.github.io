---
title: "Machine Learning Yearning 정리(1)"
categories:
  - machine learning
---

Andrew Ng의 Machine Learning Yearning 정리(1)
============

앞으로 스스로 공부한 내용을 공부하고 정리하기 위해서 글로 남기고자 합니다. 첫 번째 글은 Andrew Ng의 Machine Learning Yearning에 대한 글입니다. 해당 책은 다음 링크에서 다운받을 수 있습니다.   


[Machine Learning Yearning PDF](https://tensorflowkorea.files.wordpress.com/2018/09/ng_mly01_13.pdf)

해당 책을 한글로 번역한 Github 링크도 첨부드립니다.

[Machine Learning Yearning 한글](https://github.com/deep-diver/Machine-Learning-Yearning-Korean-Translation)




책 소개
------------

- 저자: Andrew Ng
- 제목: Machine Learning Yearning
  - 한국어로, Yearning은 '갈망' 정도로 해석할 수 있습니다.
- 목차
  - Setting up development and test sets
  - Basic Error Analysis
  - Bias and Variance
  - Learning curves
  - Comparing to human-level performance
  - Training and testing on different distributions
  - Debugging inference algorithms
  - End-to-end deep learning
  - Error analysis by parts  

  위의 챕터는 목차는 대분류이고 실제 전체 챕터는 58챕터입니다.


책 특징
------------
이 책은 Andrew Ng의 Coursera, [Machine Learning](https://www.coursera.org/learn/machine-learning) 수업을 수강한 정도의 배경 지식을 가지고 있는 사람을 위해서 쓰여졌습니다. 이 책에서는 전혀 이론을 다루지 않습니다. 대신 실제 프로젝트 시에 마주치는 문제들에 대한 해결 Tip을 담고 있습니다.

하나의 예를 들어서 어떤 내용을 다루는지 설명드리겠습니다. 우리가 neural network를 학습할 때 다음 두 가지 경우를 만나면 어떻게 대처해야 할까요?


1. Case 1
  - Training error = 1%
  - Validation error = 11%

2. Case 2
  - Training error = 15%
  - Dev error = 16%

Case 1은 Overfitting이 되어있으므로 bias보다는 variance가 높은 상황입니다. 이와 같은 경우에는는 training set에 더 데이터를 추가해야 합니다.

Case 2는 Underfitting이 된 경우므로 bias가 높은 상황입니다. 이와 같은 경우에는 neural network의 layer을 더 깊게 해서 문제를 해결해야 합니다.

Machine Learning Yearning에서는 위와 같은 실무 팁을 담고 있습니다. 따라서 이론을 공부하시는 분들보다는 실제로 프로젝트를 진행하고 계신 분이 보시면 많은 도움이 될 것 같습니다.



책 정리
----------
해당 책의 내용을 정리할 겸 앞으로 목차 중 대분류마다 하나씩 정리를 해나가보겠습니다.
