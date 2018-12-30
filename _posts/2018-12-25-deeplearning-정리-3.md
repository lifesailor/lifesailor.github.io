---
title: 딥러닝 정리(3) - Deep Neural Network
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 구현과 함께 정리된 곳은 많지 않아서 구현을 중심으로 정리할 생각입니다. Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했습니다. 

<br/>

![](/assets/images/deep-learning/deep-neural-network/dnn.png)

<br/>

Deep Neural Network은 층을 여러개 쌓은 네트워크입니다. 모델을 정의하기 전까지는 층의 개수가 정해져있지 않으므로 Two Layer Network처럼 W1, b1처럼 각각의 파라미터를 명시해서 구현할 수는 없습니다. 위의 그림에서 보듯이 Two Layer Network와 순전파와 역전파 하는 형태는 같지만 Layer 개수에 따라서 순전파와 역전파를 하는 횟수가 달라지도록 구현해야 합니다.

<br/>

따라서 구현 시에는 층의 개수를 바탕으로 반복문을 돌면서 순전파와 역전파를 구현합니다. 이 때 조심할 점은 마지막 층의 경우에는 sigmoid 함수를 사용하므로 별도로 구현해주어야 한다는 점입니다. Deeplearning book 6장 중에 위 과정이 잘 요약되어 있는 부분이 첨부합니다.

![](/assets/images/deep-learning/deep-neural-network/goodfellow-1.png)

![](/assets/images/deep-learning/deep-neural-network/goodfellow-2.png)

<br/>

아래는 Deep Neural Network를 구현한 코드입니다. 한글로 주석을 달아두었습니다. 혹시 궁금하신 점이나 잘못된 부분이 있다면 말씀해주세요.

```python
"""
main.py
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from util import relu, relu_backward, sigmoid, sigmoid_backward


np.random.seed(2018)


class DeepNeuralNetwork(object):
    def __init__(self, layers=[None, 128, 128, 1]):

        # 파라미터를 저장할 딕셔너리 - W, b
        self.parameters = {}

        # 기울기를 저장할 딕셔너리 - dW, db
        self.grads = {}

        # 노드 값을 저장할 딕셔너리 - Z, A
        self.caches = {}

        # 전체 레이어 개수
        self.layers = layers

    def initialize_weight(self, X):
        """
        파라미터 초기화

        :param X: (데이터 개수, 데이터 차원)
        :return: None
        """
        assert X.ndim == 2

        for i in range(1, len(self.layers)):
            self.parameters['W' + str(i)] = np.random.randn(self.layers[i-1], self.layers[i]) * 0.01
            self.parameters['b' + str(i)] = np.random.randn(1, self.layers[i]) * 0.01

    def forward(self, X):
        """
        순전파

        :param X: (데이터 개수, 데이터 차원)
        :return: Last Activation
        """
        assert X.shape[1] == self.parameters['W1'].shape[0]

        # 은닉층 순전파
        self.caches['A0'] = X
        for i in range(1, len(self.layers) - 1):
            Z = np.dot(self.caches['A' + str(i-1)], self.parameters['W' + str(i)]) + self.parameters['b' + str(i)]
            A = relu(Z)

            self.caches['Z' + str(i)] = Z
            self.caches['A' + str(i)] = A

        # 마지막 층 순전파
        ZL = np.dot(self.caches['A' + str(i)], self.parameters['W' + str(i+1)]) + self.parameters['b' + str(i+1)]
        AL = sigmoid(ZL)

        self.caches['Z' + str(i+1)] = ZL
        self.caches['A' + str(i+1)] = AL

        return AL

    def compute_cost(self, AL, Y):
        """
        cost 계산

        :param AL: 마지막 활성화 값(A2)
        :param Y: 정답
        :return:
        """

        assert AL.shape[0] == Y.shape[0]
        m = Y.shape[0]

        # 오차
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

        # 레이어 층의 수
        layers_dim = len(self.layers)

        # 마지막 층 역전파
        dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ = sigmoid_backward(self.caches['Z' + str(layers_dim - 1)], dA)

        # 은닉층 역전파
        for i in reversed(range(1, len(self.layers))):
            dW = 1 / m * np.dot(self.caches['A' + str(i-1)].T, dZ)
            db = 1 / m * np.sum(dZ, axis=0, keepdims=True)

            self.grads['dW' + str(i)] = dW
            self.grads['db' + str(i)] = db

            if i == 1: break

            dA = np.dot(dZ, self.parameters['W' + str(i)].T)
            dZ = relu_backward(self.caches['Z' + str(i - 1)], dA)

    def update_parameter(self, learning_rate=0.01):
        """
        파라미터 업데이트

        :param learning_rate: 학습률
        :return:
        """

        # 층의 개수
        layers_dim = len(self.layers)

        for i in range(1, layers_dim):
            self.parameters['W' + str(i)] -= learning_rate * self.grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= learning_rate * self.grads['db' + str(i)]

    def train(self, X, Y, learning_rate=0.01):
        """
        1) 순전파
        2) 오차 함수 계산
        3) 역전파
        4) 파라미터 업데이트

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
    예측 확률을 [0, 1]로 반올림

    :param AL: 마지막 층 활성화 값
    :return:
    """
    prediction = AL.copy()
    prediction[AL >= 0.5] = 1
    prediction[AL < 0.5] = 0

    return prediction


def caculate_accuracy(AL, Y):
    """
    평균 정확도 계산

    :param AL: 마지막 층 활성화 값
    :param Y: 정답 레이블
    :return:
    """
    return np.average(AL == Y)


if __name__ == "__main__":

    # 1. sklearn 에서 데이터 불러오기
    data = load_breast_cancer()
    X, y = data['data'], data['target']
    y = y.reshape(-1, 1)

    # 2. 전처리
    # train test 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # 각 특성별 크기를 [0, 1]로 바꾼다.
    scaler = MinMaxScaler()
    preprocssed_X_train = scaler.fit_transform(X_train)
    preprocssed_X_test = scaler.transform(X_test)

    # 3. 훈련
    # 모델 정의
    deep_neural_network = DeepNeuralNetwork(layers=[X_train.shape[1], 32, 32, 1])

    # 파라미터 초기화
    deep_neural_network.initialize_weight(X)

    # 하이퍼파라미터 세팅
    BATCH_SIZE = 32
    EPOCHS = 1000
    LEARNING_RATE = 0.01

    train_history = []
    train_acc = []

    test_history = []
    test_acc = []

    # 학습
    for i in range(EPOCHS):
        # 에폭 당 배치 개수
        num_iterations = X_train.shape[0] // BATCH_SIZE
        epoch_cost = 0

        # 배치 학습
        for j in range(num_iterations):
            X_train_batch = preprocssed_X_train[j * BATCH_SIZE:(j+1) * BATCH_SIZE, :]
            y_train_batch = y_train[j * BATCH_SIZE:(j+1)*BATCH_SIZE, 0].reshape(-1, 1)

            if j == num_iterations - 1:
                X_train_batch = preprocssed_X_train[j * BATCH_SIZE:, :]
                y_train_batch = y_train[j * BATCH_SIZE:, 0].reshape(-1, 1)

            cost = deep_neural_network.train(X_train_batch, y_train_batch,
                                             learning_rate=LEARNING_RATE)
            epoch_cost += cost

        # 에폭 평균 오차
        epoch_cost /= X_train.shape[0]

        if i % 10 == 0:
            print(str(i+1) + " 번째 cost : ", epoch_cost)

        # train 오차 및 정확도
        train_history.append(epoch_cost)
        prediction = predict(deep_neural_network.forward(preprocssed_X_train))
        acc = caculate_accuracy(prediction, y_train)
        train_acc.append(acc)

        # test 오차 및 정확도
        test_AL = deep_neural_network.forward(preprocssed_X_test)
        cost = deep_neural_network.compute_cost(test_AL, y_test)
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
    relu 함수 구현 - A = relu(Z)

    :param X:
    :return: relu(X)
    """

    A = np.copy(X)
    A[A < 0] = 0
    return A


def relu_backward(Z, dA):
    """
    relu 역전파 구현 - dZ = relu'(Z) * dA

    :param Z:
    :param dA:
    :return: dZ
    """

    relu_derivative = np.copy(Z)
    relu_derivative[Z < 0] = 0
    relu_derivative[Z > 0] = 1
    dZ = relu_derivative * dA

    return dZ


def sigmoid(X):
    """
    sigmoid 함수 구현 - A = sigmoid(X)

    :param X:
    :return: A
    """
    A = 1 / ((1 + np.exp(-X)) + 1e-5)
    return A


def sigmoid_backward(Z, dA):
    """
    sigmoid 역전파 구현 - dZ = sigmoid'(Z) * dA


    :param Z:
    :param dA:
    :return: dZ
    """

    AL = sigmoid(Z)
    sigmoid_derivative = AL * (1 - AL)
    dZ = sigmoid_derivative * dA
    return dZ
```

