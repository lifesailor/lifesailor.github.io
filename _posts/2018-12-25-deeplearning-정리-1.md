---
title: 딥러닝 정리(1) - Logistic Regression
categories:
  - deep learning
tags:
  - machine learning
  - deep learning

---

앞으로 딥러닝을 공부하면서 하나씩 정리해보고자 합니다. 딥러닝 개념은 정리된 블로그는 많지만 코드가 정리된 곳은 많지 않다고 느껴서 구현 위주로 정리하고자 합니다. 정리와 코드는 Andrew Ng 교수님의 [deeplearning.ai](https://www.coursera.org/courses?query=deeplearning.ai) 강의를 참고했습니다. 혹시 잘못된 부분이 있으면 말씀해주세요. 



Logistic Regression은 어떤 사건 A가 일어날 확률을 fitting하는 것입니다. 확률 P는 [0, 1] 범위 안에 속하기에 Linear Regression을 바로 적용하기에 부적합니다. 그래서 logit이라는 개념을 도입해서 확률 P를 [0, 1] 사이의 값으로 변환해서 fitting을 합니다.

Logistic Regression 구현 시 어려운 부분은 역전파 부분입니다. 아래는 Logistic Regression의 역전파 과정입니다. 



<br/>

$odds = p(y = 1)/p(y = 0|x) = p(y=1|x)/(1-p(y=1|x))$

$logit(p) = log(p(y=1|x))/p(1-p(y=1|x)) = w^tx$



logit(p) 식을 p(y = 1|x)를 기준으로 정리하면 

$p(y=1|x) = e^{w^{t}x}/(1 + e^{w^{t}x}) = 1 / (1+ e^{-w^{t}x})$ 

이고 위 식이 익숙한 sigmoid 함수입니다.

<br/>



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

아래는 Logistic Regression 구현한 코드입니다.

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

# 3. forware propagation 함수 구현
def forward(X, Y, W, b):
    # linear - Z = XW + b
    Z = np.dot(X, W) + b
    
    # sigmoid activation
    A = sigmoid(Z)
    
    # cost
    m = X.shape[0]
    
    # cross entropy loss
    cost = -1/m * np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)))
    return A, cost

# 4. forward - backward propagation
def propagate(X, Y, W, b, learning_rate=0.001):
    m = X.shape[0]
    
    # forward
    A, cost = forward(X, Y, W, b)
    
    # backward - dZ = dL/dA * dA/dZ = (A - Y)
    dw = 1/m * np.dot(X.T, A - Y)  
    db = 1/m * np.sum(A - Y)
    
    # gradient update
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

# 6. Test
def predict(X, Y, W, b):
    prediction, cost = forward(X, Y, W, b)
    actual = Y
    
    # 예측 확률을 [0, 1]로 반올림한다.
    predicted_class = np.zeros((prediction.shape[0], 1))
    
    for i in range(prediction.shape[0]):
        if prediction[i, 0] > 0.5:
            predicted_class[i, 0] = 1
        else:
            predicted_class[i, 0] = 0
    return predicted_class

# 7. Result
# train
prediction = predict(X_train, y_train, W, b)
np.mean(prediction == y_train)

# test
prediction = predict(X_test, y_test, W, b)
np.mean(prediction == y_test)
```