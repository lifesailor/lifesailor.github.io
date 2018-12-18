---
title: 딥러닝 정리(1) - Logistic Regression
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 정리 순서는 Andrew Ng의 [deeplearning.ai](https://www.coursera.org/specializations/deep-learning)와 Standford의 [CS231n](http://cs231n.stanford.edu/) [CS224n](http://cs224d.stanford.edu/)를 참고했습니다.



Logistic Regression은 확률을 fitting하는 것입니다. 하지만 확률 p는 [0, 1] 범위 안에 속하기에 Linear Regression을 바로 적용하기에 부적절했고 logit이라는 개념을 도입해서 p를 [0, 1] 변환해서 fitting한 뒤에 fitting을 합니다. 다음이 그 절차입니다.

$$odds = p(y=1) / p(y=0|x) = p(y=1|x) / (1 - p(y=1|x))$$

$$logit(p) = log(p(y=1|x)) / (1 - p(y=1|x)) = w^Tx$$

$$logitstic function = e^{w^Tx} / (1 + e^{w^Tx} ) = 1 / (1 + e^{-w^Tx})$$

[Logistic Regression](https://ko.wikipedia.org/wiki/%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1_%ED%9A%8C%EA%B7%80)



머신러닝을 배울 때는 logistic regression을 sigmoid 함수(logistic 함수)를 적용해서 분류 문제를 해결하는 방법으로 소개됩니다. Logistic Regression 학습 과정은 여러가지 최적화 방법을 사용할 수 있지만 딥러닝의 전 단계로 logistic regression을 배울 때는 gradient descent 방법으로 최적화를 합니다. 비용 함수는 Maximum Likelihood Estimator에 기반한 cross entropy loss를 사용합니다.



Logistic Regression 구현 코드는 다음과 같습니다.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

### 1. 데이터
# sklearn에서 데이터 불러오기
data = load_breast_cancer()
X, y = data['data'], data['target']

# Train, Test로 데이터를 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

### 2. Logistic Regression
### Training
# 1. sigmoid function 정의
def sigmoid(x):
    return 1 / ((1 + np.exp(-x)) + 1e-5)

# 2. weight 초기화
def weight_initializer(X):
    W = np.zeros(X.shape[1]).reshape(-1, 1)
    b = 0.0     
    return W, b

# weight initializer
W, b = weight_initializer(X_train)

# 3. forward propagate
def forward(X, Y, W, b):
    # linear
    Z = np.dot(X, W) + b
    
    # activation
    A = sigmoid(Z)
    
    # cost
    m = X.shape[0]
    cost = -1/m * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)))
    return A, cost

# 4. propagate - forward, backward
def propagate(X, Y, W, b, learning_rate=0.001):
    m = X.shape[0]
    
    # forward
    A, cost = forward(X, Y, W, b)
    
    # backward
    dw = 1/m * np.dot(X.T, A - Y)
    db = 1/m * np.sum(A - Y)
    
    assert(dw.shape == W.shape)
    
    # update
    W = W - learning_rate * dw
    b = b - learning_rate * db

    return W, b, cost   

y_train = y_train.reshape(-1, 1)

# 5. training
EPOCH = 10000
costs = []

W, b = weight_initializer(X_train)
for i in range(EPOCH):
    W, b, cost = propagate(X_train, y_train, W, b, learning_rate=0.0001)
    if i % 1000 == 0:
        print(cost)

### Test
def predict(X, Y, W, b):
    prediction, cost = forward(X, Y, W, b)
    actual = Y
    
    predicted_class = np.zeros((prediction.shape[0], 1))
    
    for i in range(prediction.shape[0]):
        if prediction[i, 0] > 0.5:
            predicted_class[i, 0] = 1
        else:
            predicted_class[i, 0] = 0
    return predicted_class

# train
prediction = predict(X_train, y_train, W, b)
np.mean(prediction == y_train)

# test
prediction = predict(X_test, y_test, W, b)
np.mean(prediction == y_test)
```





