---
title: 딥러닝 정리(2) - Two Layer Network
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. 정리와 코드는 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의를 참고했습니다. 

<br/>

![](/assets/images/deep-learning/two-layer-network/network.png)

<br/>

Two Layer Network은 Logistic Regression과 한 층을 더 쌓았다는 점에서 다릅니다. 그래서 순전파를 2번 하고 역전파도 2번 합니다. 위 그림에서 선형 결합과 활성화 함수를 구분지어서 표현해서 층이 더 많아보이지만, (Z1, A1), (Z2, A2), 총 2층의 Neural Network입니다. 

<br/>

Two Layer Network 역전파 구현 시 Logistic Regression 구현과 달라지는 부분은 위 그림에서 빨간색 글씨로 적은 dL/dA1, dL/dZ1 부분입니다. 은닉 층의 활성화 함수에 대한 역전파를 하기 때문입니다. 해당 부분의 수식과 코드는 다음과 같습니다.

<br/>

$dL/dA1 = dL/dZ2 * dZ2 / dA1 = dL/dZ2 * {W2}^T$

$dL/dZ1 = dL / dA1 * dA1/dZ1 = relu'(Z1) * dL/dA1$

```python
# 은닉층 역전파
dA1 = np.dot(dZ2, self.parameters['W2'].T)
dZ1 = relu_backward(self.caches['Z1'], dA1)

def relu_backward(Z, dA):
    """
    relu backward

    :param Z:
    :param dA:
    :return: dZ = relu'(Z) * dA
    """

    relu_derivative = np.copy(Z)
    relu_derivative[Z < 0] = 0
    relu_derivative[Z > 0] = 1
    dZ = relu_derivative * dA

    return dZ
```

<br/>

역전파 과정 외의 나머지 부분은 위의 그림의 수식을 그대로 구현하면 됩니다. 아래 train 함수를 보면 전체 학습 과정이 정리되어 있습니다. 네트워크는 순전파, 오차함수 계산, 역전파, 파라미터 업데이트를 반복하면서 가중치를 학습해나갑니다.

```python
"""
main.py
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from util import relu, relu_backward, sigmoid, sigmoid_backward

np.random.seed(42)


class TwoLayerNetwork(object):
    def __init__(self, hidden_dim=128):
        # W1, b1, W2, b2
        self.parameters = {}

        # dW1, db1, dW2, db2
        self.grads = {}

        # X, Z1, A1, Z2, A2
        self.caches = {}

        # hidden dim
        self.hidden_dim = hidden_dim

    def initialize_weight(self, X):
        """
        weight 초기화

        :param X: (데이터 개수, 데이터 차원)
        :return: None
        """
        assert X.ndim == 2

        input_dim = X.shape[1]

        W1 = np.random.randn(input_dim, self.hidden_dim) * 0.01
        W2 = np.random.randn(self.hidden_dim, 1) * 0.01
        b1 = np.random.randn(1, self.hidden_dim) * 0.01
        b2 = np.random.randn(1, 1) * 0.01

        self.parameters['W1'] = W1
        self.parameters['b1'] = b1
        self.parameters['W2'] = W2
        self.parameters['b2'] = b2

    def forward(self, X):
        """
        순전파

        :param X: (데이터 개수, 데이터 차원)
        :return: Last Activation
        """
        assert X.shape[1] == self.parameters['W1'].shape[0]

        Z1 = np.dot(X, self.parameters['W1']) + self.parameters['b1']
        A1 = relu(Z1)
        Z2 = np.dot(A1, self.parameters['W2']) + self.parameters['b2']
        A2 = sigmoid(Z2)

        self.caches['A0'] = X
        self.caches['Z1'] = Z1
        self.caches['A1'] = A1
        self.caches['Z2'] = Z2
        self.caches['A2'] = A2

        return A2

    def compute_cost(self, AL, Y):
        """
        cost 계산

        :param AL: 마지막 활성화 값(A2)
        :param Y: 정답
        :return:
        """

        assert AL.shape[0] == Y.shape[0]
        m = Y.shape[0]
        cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        return cost

    def backward(self, AL, Y):
        """
        역전파 구현

        :param AL: 마지막 활성화 값(A2)
        :param Y:
        :return:
        """

        m = Y.shape[0]

        dA2 = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ2 = sigmoid_backward(self.caches['Z2'], dA2)

        dW2 = 1 / m * np.dot(self.caches['A1'].T, dZ2)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.parameters['W2'].T)
        dZ1 = relu_backward(self.caches['Z1'], dA1)

        dW1 = 1 / m * np.dot(self.caches['A0'].T, dZ1)
        db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1

    def update_parameter(self, learning_rate=0.01):
        """
        파라미터 업데이트

        :param learning_rate: 학습률
        :return:
        """
        for i in range(1, 3):
            self.parameters['W' + str(i)] -= learning_rate * self.grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= learning_rate * self.grads['db' + str(i)]

    def train(self, X, Y, learning_rate=0.01):
        """
        1) 순전파
        2) 역전파
        3) 파라미터 업데이트

        :param X: 데이터
        :param Y: 정답 레이블
        :param learning_rate: 학습률
        :return:
        """

        AL = self.forward(X)
        cost = self.compute_cost(AL, Y)
        self.backward(AL, Y)
        self.update_parameter(learning_rate=learning_rate)
        return cost


def predict(AL):
    """

    :param AL:
    :return:
    """
    prediction = AL.copy()
    prediction[AL >= 0.5] = 1
    prediction[AL < 0.5] = 0

    return prediction


def caculate_accuracy(AL, Y):
    return np.average(AL == Y)


if __name__ == "__main__":

    # 1. sklearn 에서 데이터 불러오기
    data = load_breast_cancer()
    X, y = data['data'], data['target']
    y = y.reshape(-1, 1)

    # 2. 전처리
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # scaling
    scaler = MinMaxScaler()
    preprocssed_X_train = scaler.fit_transform(X_train)
    preprocssed_X_test = scaler.transform(X_test)

    # 3. 훈련

    # 모델 정의
    two_layer_network = TwoLayerNetwork(hidden_dim=128)

    # 파라미터 초기화
    two_layer_network.initialize_weight(X)

    # hyperparameter
    BATCH_SIZE = 32
    EPOCHS = 1000
    LEARNING_RATE = 0.01

    train_history = []
    train_acc = []

    test_history = []
    test_acc = []

    # train model
    for i in range(EPOCHS):
        num_iterations = preprocssed_X_train.shape[0] // BATCH_SIZE
        epoch_cost = 0

        for j in range(num_iterations):
            X_train_batch = preprocssed_X_train[j * BATCH_SIZE:(j+1) * BATCH_SIZE, :]
            y_train_batch = y_train[j * BATCH_SIZE:(j+1)*BATCH_SIZE, 0].reshape(-1, 1)

            if j == num_iterations - 1:
                X_train_batch = X_train[j * BATCH_SIZE:, :]
                y_train_batch = y_train[j * BATCH_SIZE:, 0].reshape(-1, 1)

            cost = two_layer_network.train(X_train_batch, y_train_batch,
                                           learning_rate=LEARNING_RATE)
            epoch_cost += cost
        epoch_cost /= X_train.shape[0]

        if i % 10 == 0:
            print(str(i+1) + " 번째 cost : ", epoch_cost)

        train_history.append(epoch_cost)
        prediction = predict(two_layer_network.forward(preprocssed_X_train))
        acc = caculate_accuracy(prediction, y_train)
        train_acc.append(acc)

        # test
        test_AL = two_layer_network.forward(preprocssed_X_test)
        cost = two_layer_network.compute_cost(test_AL, y_test)
        test_history.append(cost)

        prediction = predict(test_AL)
        acc = caculate_accuracy(prediction, y_test)
        test_acc.append(acc)

    # 4. 출력
    plt.plot(train_history, label='train')
    plt.plot(test_history, label='test')
    plt.xlabel("EPOCH")
    plt.ylabel("COST")
    plt.legend()
    plt.show()

    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.xlabel("EPOCH")
    plt.ylabel("ACC")
    plt.legend()
    plt.show()

```

```python
"""
util.py
"""

import numpy as np

def relu(X):
    """
    relu function

    :param X:
    :return: relu(X)
    """

    A = np.copy(X)
    A[A < 0] = 0
    return A


def relu_backward(Z, dA):
    """
    relu backward

    :param Z:
    :param dA:
    :return: dZ = relu'(Z) * dA
    """

    relu_derivative = np.copy(Z)
    relu_derivative[Z < 0] = 0
    relu_derivative[Z > 0] = 1
    dZ = relu_derivative * dA

    return dZ


def sigmoid(X):
    """
    sigmoid function

    :param X:
    :return: sigmoid(X)
    """

    return 1 / ((1 + np.exp(-X)) + 1e-5)


def sigmoid_backward(Z, dA):
    """
    sigmoid backward


    :param Z:
    :param dA:
    :return: dZ = sigmoid'(Z) * dA
    """

    AL = sigmoid(Z)
    sigmoid_derivative = AL * (1 - AL)
    dZ = sigmoid_derivative * dA
    return dZ
```

