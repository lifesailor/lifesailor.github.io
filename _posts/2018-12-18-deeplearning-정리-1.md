---
title: 딥러닝 정리(1) - Logistic Regression
categories:
  - deep learning
tags:
  - machine learning
  - deep learning
---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 코드가 정리된 곳은 많지 않다고 느껴서 코드 위주로 정리하고자 합니다. 코드에는 한글 주석을 달아두었습니다. 



Logistic Regression은 확률을 fitting하는 것입니다. 하지만 확률 p는 [0, 1] 범위 안에 속하기에 Linear Regression을 바로 적용하기에 부적절했고 logit이라는 개념을 도입해서 p를 [0, 1] 변환해서 fitting한 뒤에 fitting을 합니다. 다음이 그 절차입니다.

![](/assets/images/logistic-regression/logistic.png)

하지만 머신러닝을 배울 때는 logistic regression을 단순히 sigmoid 함수(logistic 함수)를 적용해서 분류 문제를 해결하는 방법으로 소개됩니다. 



다음은 Logistic Regression 구현 코드입니다.

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
    
    # update
    W = W - learning_rate * dw
    b = b - learning_rate * db

    assert(dw.shape == W.shape)
    
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





