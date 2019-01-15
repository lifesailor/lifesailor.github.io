---
title: 딥러닝 정리(4) - Weight Initialization
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

신경망의 가중치를 어떻게 초기화하느냐에 따라서 성능이 많이 달라집니다. 하나의 예로 초기 가중치를 각 층마다 같게 대칭으로 설정하면 모든 신경망의 가중치가 같게 순전파 되고 역전파도 동일하게 됩니다. 따라서 계속해서 각각의 노드가 출력하는 값이 같게 되고 신경망은 제대로 작동하지 못합니다.

<br/>

![](/assets/images/deep-learning/initialization/intialization.png)

<br/>

위의 신경망 한 층을 봅시다. $Z = w1x1 + w2x2 + \dots + wnxn​$ 입니다. 만약에 n이 커지면 어떻게 될까요? 점점 더 많은 항의 합이 되기 때문에 Z가 너무 커지거나 작아질 수 있습니다. 이는 순전파와 역전파 과정에서 값들이 폭발할 수 있습니다. 이를 방지하기 위해 노드의 개수에 따라서 초기화를 다르게 하는 방법들이 고안되었습니다. 그 중 가장 유명한 것이 XIaver와 He의 가중치 초기화 방법입니다.

```python
# Xiavier initialization
W = np.random.randn(input_dim, output_dim) * np.sqrt(1/input_dim)

# He initialization
W = np.random.randn(input_dim, output_dim) * np.sqrt(2/input_dim)
```

두 가지 방법 모두 입력 노드의 수에 맞게 가중치의 분산을 $Var(w) = \frac{1}{n}$ 또는 $Var(w) = \frac{2}{n}$으로 제어합니다. Xiavier initialization은 tanh 활성 함수에 잘 작동하고 He initialization은 relu 활성 함수에 잘 작동한다고 알려져 있습니다.



