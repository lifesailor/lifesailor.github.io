---
title: "Signal Processing(1) - Introduction"
categories:
  - signal processing
tags:
  - signal porcessing
---

회사에서 진동 데이터로 장비 고장을 예지하는 업무를 하고 있습니다. 신호 처리는 진동 데이터를 분석하기 위한 초석이 됩니다. 생소한 분야지만 하나씩 정리하면서 스스로 배움을 얻고자 합니다. [Coursera](https://www.coursera.org/learn/dsp)와  [Udemy](https://www.udemy.com/signal-processing/)의 신호처리 강의를 참고했습니다. 

<br/>

### 1. Signal Processing 정의

- Signal 
  - 물리적인 현상에 대한 진화에 대한 설명
    - 날씨를 온도로 설명한다.
    - 소리를 압력으로 설명한다.
    - 빛을 명도로 표현한다.

- Processing 
  - 크게 Analysis(분석)과 Synthesis(합성)이 있다.
    - Analysis(분석) - 신호의 정보를 이해한다.
    - Synthesis(합성) - 주어진 정보를 포함하는 신호를 생성한다.

<br/>

### 2. Digital Signal Processing

기존의 Analog Signal를 Digital Signal로 처리할 수 있게 되면서 신호 처리 분야의 혁명을 가져왔다.

- Discretization of time
  - sample replace idealized models
  - sample math replaces calculus



- Discretization of values
  - general purpose storage
  - general purpose processing
  - noise can be controlled

<br/>

### 3. Digital Time signal

- Digital Time Signal은 복소수의 나열이다.

- 4가지 신호 유형

  - Finite-length

    - $x[n] = [x_0, x_1, x_{n-1}]^t$

  - Infinite-length

    - Finite-length의 확장

  - Periodic

    - $x[n] = x[n + kN]$

  - Finite-support


- Energy and Power
  - Energy = $E_x = \sum_{n=-\infty}^{+\infty}{x[n]}^2$
  - Power = $P_x = \lim_{n\to \infty}\frac{1}{2N+1}E_x$

<br/>

### 4.  Digital vs Physical Frequency

- Discrete Time
  - no physical dimension
  - periodicity: how many samples before pattern repeats



- Physical world
  - periodicity: how many seconds before pattern repeats
  - frequency: measurd in Hz($s^{-1}$)



- Discrete-Physical Bridge
  - Ts: Time between samples
  - M: periodicty of MY seconds
  - Physical world frequency $f = 1/(M * Ts)$
    - 한마디로 진동수는 1/주기이라는 말이다.

<br/>

### 5. Karplus Strong Algorithm





The Karplus-Strong Alogirhtm is a simple digital feedback loop with an internal buffer of M samples.

```python
def KS(x, N, alpha=0.99):
    """
    Karplus-Strong-algoritm
    
    x: standard signal
    N: length of new signal
    alpha: controls envelope(decay)
    
    return: y: new signal
    """
    M = len(x) # M controls frequency
    y = np.zeros(N) # 
    
    for n in range(0, N):
        y[n] = (x[n] if n < M else 0) + alpha * (y[n-M] if n-M >=0 else 0)
        
    return y
```

