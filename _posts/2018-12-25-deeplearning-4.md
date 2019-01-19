---
title: Deep Learning 정리(4) - Two Layer Network
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

1-2. Two Layer Network

<br/>

![](/assets/images/deep-learning/two-layer-network/network.png)

<br/>

'딥러닝이란 무엇이고 어떻게 동작 하는가.'의 Part2은 Two Layer Network입니다. Two Layer Network은 Logistic Regression과 한 층을 더 쌓았다는 점에서 다릅니다. 그래서 순전파를 2번 하고 역전파도 2번 합니다. 위 그림에서 선형 결합과 활성화 함수를 구분지어서 표현해서 층이 더 많아보이지만, (Z1, A1), (Z2, A2), 총 2층의 Neural Network입니다. 

<br/>

Two Layer Network 역전파 구현 시 Logistic Regression 구현과 달라지는 부분은 위 그림에서 빨간색 글씨로 적은 dL/dA1, dL/dZ1 부분입니다. 은닉 층의 활성화 함수에 대한 역전파를 하기 때문입니다. 해당 부분의 수식은 다음과 같습니다.

<br/>

$dL/dA1 = dL/dZ2 * dZ2 / dA1 = dL/dZ2 * {W2}^T$

$dL/dZ1 = dL / dA1 * dA1/dZ1 = relu'(Z1) * dL/dA1$

<br/>

아래는 Two Layer Network를 구현한 코드입니다. 한글로 주석을 달아두었습니다. 혹시 궁금하신 점이나 잘못된 부분이 있다면 말씀해주세요.

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

        # 파라미터를 저장할 딕셔너리 - W1, b1, W2, b2
        self.parameters = {}

        # 기울기를 저장할 딕셔너리 - dW1, db1, dW2, db2
        self.grads = {}

        # 은닉층 노드 값을 저장할 딕셔너리 - X, Z1, A1, Z2, A2
        self.caches = {}

        # 은닉 층 노드 개수
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
        :param Y: 정답 레이블
        :return: cost
        """

        assert AL.shape[0] == Y.shape[0]
        m = Y.shape[0]
        cost = - 1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        return cost

    def backward(self, AL, Y):
        """
        역전파 구현

        :param AL: 마지막 활성화 값(A2)
        :param Y: 실제 Label
        :return: None
        """

        m = Y.shape[0]

        # 최종 오차 역전파
        dA2 = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ2 = sigmoid_backward(self.caches['Z2'], dA2)

        dW2 = 1 / m * np.dot(self.caches['A1'].T, dZ2)
        db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

        # 활성화 함수 역전파
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
        :return: None
        """
        for i in range(1, 3):
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
    :return: prediction(0 또는 1)
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
    :return: 정확도
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
    two_layer_network = TwoLayerNetwork(hidden_dim=128)

    # 파라미터 초기화
    two_layer_network.initialize_weight(X)

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
        num_iterations = preprocssed_X_train.shape[0] // BATCH_SIZE
        epoch_cost = 0

        # 배치 학습
        for j in range(num_iterations):
            X_train_batch = preprocssed_X_train[j * BATCH_SIZE:(j+1) * BATCH_SIZE, :]
            y_train_batch = y_train[j * BATCH_SIZE:(j+1)*BATCH_SIZE, 0].reshape(-1, 1)

            if j == num_iterations - 1:
                X_train_batch = X_train[j * BATCH_SIZE:, :]
                y_train_batch = y_train[j * BATCH_SIZE:, 0].reshape(-1, 1)

            cost = two_layer_network.train(X_train_batch, y_train_batch,
                                           learning_rate=LEARNING_RATE)
            epoch_cost += cost

        # 에폭 평균 오차
        epoch_cost /= X_train.shape[0]

        if i % 10 == 0:
            print(str(i+1) + " 번째 cost : ", epoch_cost)

        # train 오차 및 정확도
        train_history.append(epoch_cost)
        prediction = predict(two_layer_network.forward(preprocssed_X_train))
        acc = caculate_accuracy(prediction, y_train)
        train_acc.append(acc)

        # test 오차 및 정확도
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
    relu 함수 구현 - A = relu(Z)

    :param X:
    :return: A
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

