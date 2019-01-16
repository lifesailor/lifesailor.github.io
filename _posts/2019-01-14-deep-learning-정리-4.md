---
title: 딥러닝 구현 정리(4) - Weight Initialization
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의 및 Standford의 [CS231n](http://cs231n.stanford.edu/) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

신경망의 가중치를 어떻게 초기화하느냐에 따라서 성능이 많이 달라집니다. 하나의 예로 초기 가중치를 각 층의 노드마다 같게 설정하면 모든 신경망의 가중치가 같게 순전파 되고 역전파도 동일하게 됩니다. 따라서 계속해서 각각의 노드가 출력하는 값이 같게 되고 신경망은 제대로 작동하지 못합니다.

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

두 가지 방법 모두 입력 노드의 수에 맞게 가중치의 분산을 $Var(w) = \frac{1}{n}$ 또는 $Var(w) = \frac{2}{n}$으로 제어합니다. 일반적으로 Xiavier 가중치 초기화 방법은 잘 작동하지만 relu 활성화 함수에는 He initialization은  잘 작동한다고 알려져 있습니다. relu 활성화 함수는 음수인 부분을 모두 0으로 만들기 때문에 입력 노드의 수의 절반만큼의 수로 가중치를 조정합니다. 다음은 가중치 초기화 방법에 따라서 각 층의 가중치를 출력하는 코드입니다.

```python
"""
util.py
"""
def relu(X):
    """
    relu 함수 구현 - A = relu(Z)

    :param X:
    :return: A
    """
    
    return np.maximum(X, 0)


def sigmoid(X):
    """
    sigmoid 함수 구현 - A = sigmoid(X)

    :param X:
    :return: A
    """
    A = 1 / ((1 + np.exp(-X)) + 1e-5)
    return A
```

```python
"""
main.py
"""
import numpy as np
from matplotlib import pyplot as plt
from util import relu, sigmoid


# 기본 값 세팅
x = np.random.randn(1000, 100)
node_num = 100
hidden_layer_size = 5

# activation 값 저장
activations = {}


for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i - 1]

    # w = np.random.randn(node_num, node_num) * 0.01 # Normal
    # w = np.random.randn(node_num, node_num) * np.sqrt(1 / node_num) # Xiavier
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)  # He
    a = np.dot(x, w)
    
    # z = sigmoid(a)
    # z = np.tanh(a)
    z = relu(a)

    activations[i] = z


# plot
plt.figure(figsize=(15, 5))

for i, a in activations.items():
    plt.subplot(1, len(activations), i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
```

다음 코드를 실행하면 Normal 분포로 가중치를 초기화한 경우에는 아래와 같이 몇 층이 지나면 활성화 함수 출력 값이 모두 0이 되어버립니다. 반면에 He 분포로 가중치를 초기화 한 경우에는 0을 제외하고 활성화 함수 출력 값이 여전히 0에서 1사이에 균등하게 분포되어 있는 것을 확인할 수 있습니다.

![](/assets/images/deep-learning/initialization/normal.png)

![](/assets/images/deep-learning/initialization/he.png)