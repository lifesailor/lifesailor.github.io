---
title: Deep Learning 정리(2) - Logistic Regression
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
  - logistic Regression
---

글 작성에 앞서 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의 및 Standford의 [CS231n](http://cs231n.stanford.edu/) 강의와 Ian Goodfellow의 [deeplearning book](http://www.deeplearningbook.org/)를 참고했다는 것을 밝힙니다. 

<br/>

Logistic Regression은 은닉층이 없는 신경망입니다. 그렇기 때문에 구현 과정에서 순전파와 역전파를 한 번씩만 해주면 됩니다.  위의 그림에서 구현 시 어려운 부분은 역전파 과정입니다.

<br/>

![](/assets/images/deep-learning/logistic-regression/logistic.png)

<br/>

$Chain Rule: dL / dZ = dL / dA * dA /dZ - (1)$

$Cross Entropy Loss:  L =- (Y * log(A) + (1-Y) * log(1- A)) - (2)$

$dL / dA = -(Y/A - (1 - Y) / (1- A)) - (3)$

$dA / dZ = A * (1 - A) - (4)$

$dL / dZ = A - Y  - (5)$ 

<br/>

역전파를 구현할 때는 cross entropy loss에 1/m에 곱하지 않은 Loss Vector를 역전파한다는 것이 중요합니다. 출력하는 Loss 값은 1/m을 곱해서 각 training example 별 평균 loss를 계산합니다.

<br/>

$Loss = -1/m * (Y*log(A) + (1-Y) *log(1-A))$

<br/>

하지만 역전파를 할 때는 Loss Vector 내에 각각의 training example의 loss 정보를 유지한 채로 역전파합니다. 그렇기 때문에 Loss Vector에는 1/m을 곱하지 않습니다. 대신 dW, db를 구할 때 1/m을 곱해서 평균 gradient를 계산하게 됩니다.

<br/>

$dL/dZ = A - Y$

$dW = 1/m * X^T(A - Y) $

$ db = 1/m * \sum_i(a_i - y_i) $

<br/>

아래는 Logistic Regression을 구현한 코드입니다. 한글로 주석을 달아두었습니다. 혹시 궁금하신 점이나 잘못된 부분이 있다면 말씀해주세요.

```python
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

np.random.seed(2018)


def sigmoid(x):
    """
    sigmoid 함수

    :param x: 입력 데이터
    :return: sigmoid(x)
    """
    return 1 / ((1 + np.exp(-x)) + 1e-5)


def weight_initializer(X):
    """
    weight 초기화

    :param X: 입력 데이터
    :return: W, b
    """
    W = np.random.randn(X.shape[1]).reshape(-1, 1) * 0.01
    b = np.random.randn(1) * 0.01
    return W, b


def forward(X, Y, W, b):
    """
    순전파

    :param X: 입력 데이터
    :param Y: 정답 레이블
    :param W: weight
    :param b: bias
    :return: A, cost
    """
    # linear - Z = XW + b
    Z = np.dot(X, W) + b

    # sigmoid
    A = sigmoid(Z)

    # 데이터 개수
    m = X.shape[0]

    # 오차
    cost = -1 / m * np.sum((Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    return A, cost


def propagate(X, Y, W, b, learning_rate=0.001):
    """
    1) 순전파 및 오차함수 계산
    2) 역전파
    3) 파라미터 업데이트

    :param X: 입력 데이터
    :param Y: 정답 레이블
    :param W: weight
    :param b: bias
    :param learning_rate: 학습률
    :return: W, b, cost
    """

    m = X.shape[0]

    # 순전파
    A, cost = forward(X, Y, W, b)

    # 역전파 - dZ = dL/dA * dA/dZ = (A - Y)
    dw = 1 / m * np.dot(X.T, A - Y)
    db = 1 / m * np.sum(A - Y)

    # 파라미터 업데이트
    W = W - learning_rate * dw
    b = b - learning_rate * db

    assert (dw.shape == W.shape)

    return W, b, cost


def predict(X, Y, W, b):
    """
    0 또는 1의 값으로 예측한다.

    :param X: 입력 데이터 
    :param Y: 정답 레이블
    :param W: weight
    :param b: bias
    :return: prediction(0 또는 1)
    """

    # 예측 확률
    prediction, cost = forward(X, Y, W, b)

    # 예측 확률을 [0, 1]로 반올림한다.
    predicted_class = np.zeros((prediction.shape[0], 1))

    for i in range(prediction.shape[0]):
        if prediction[i, 0] > 0.5:
            predicted_class[i, 0] = 1
        else:
            predicted_class[i, 0] = 0
    return predicted_class


def calculate_accuracy(prediction, Y):
    """
    평균 정확도 계산

    :param prediction: 예측 레이블 
    :param Y: 정답 레이블
    :return: 정확도
    """
    return np.mean(prediction == Y)


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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. 훈련
    # 파라미터 초기화
    W, b = weight_initializer(X_train)
    y_train = y_train.reshape(-1, 1)

    # 하이퍼 파라미터 세팅
    BATCH_SIZE = 32
    EPOCH = 1000
    LEARNING_RATE = 0.01

    train_history = []
    test_history = []

    train_acc = []
    test_acc = []

    # 학습
    for i in range(EPOCH):
        # 에폭 당 배치 개수
        num_iterations = X_train.shape[0] // BATCH_SIZE
        epoch_cost = 0

        # 배치 학습
        for j in range(num_iterations):
            X_train_batch = X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, :]
            y_train_batch = y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, 0].reshape(-1, 1)

            if j == num_iterations - 1:
                X_train_batch = X_train[j * BATCH_SIZE:, :]
                y_train_batch = y_train[j * BATCH_SIZE:, 0].reshape(-1, 1)

            W, b, cost = propagate(X_train_batch, y_train_batch, W, b, learning_rate=LEARNING_RATE)
            epoch_cost += cost

        # 에폭 평균 오차
        epoch_cost /= X_train.shape[0]

        if i % 10 == 0:
            print(str(i + 1) + " 번째 cost : ", epoch_cost)

        # train 오차 및 정확도
        train_history.append(epoch_cost)
        train_prediction = predict(X_train, y_train, W, b)
        train_acc.append(calculate_accuracy(train_prediction, y_train))

        # test 오차 및 정확도
        _, test_cost = forward(X_test, y_test, W, b)
        test_history.append(test_cost)
        test_prediction = predict(X_test, y_test, W, b)
        test_acc.append(calculate_accuracy(test_prediction, y_test))

    # 4. 출력
    plt.plot(train_history, label="train")
    plt.plot(test_history, label='test')
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend()
    plt.show()

    plt.plot(train_acc, label="train")
    plt.plot(test_acc, label='test')
    plt.xlabel("EPOCH")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
```

