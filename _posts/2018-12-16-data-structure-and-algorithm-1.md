---
title: "Data Structure and Algorithm(1) - Asymptotic Analysis"
categories:
  - data structure and algorithm
tags:
  - data structure
  - algorithm
---
자료구조와 알고리즘 공부를 시작했습니다. Coursera의 [San Diego](https://www.coursera.org/specializations/data-structures-algorithms) 대학 강의 커리큘럼에 맞게 정리할 생각입니다. 하지만 처음 Asymptotic Analysis의 경우에는 [Standford]() 강의를 참고 했습니다.



# 알고리즘 성능 분석



## 1. 알고리즘 분석의 원칙

- Worst 케이스를 기준으로 삼는다. (때로는 Average)
- 낮은 차수 및 상수 Term은 무시한다.
- 점근적 분석을 한다. 즉, 큰 숫자의 N에 따른 Running Time에 집중한다.





## 2. 점근적 분석

- 낮은 차수 및 상수 Term은 무시한다.
  - 상수 Term은 시스템에 따라서 다르다.
  - 낮은 차수 Term은 큰 숫자가 입력될 경우 알고리즘 성능과 무관하다.





## 3. 점근적 표기법

#### 1. Big-O Notation

- 정의
  -  $T(n) = O (f(n)) $ if and only if there exist constant $c, n_0$ such that $T(n) <= c f(n)$ for all $n >= n_0$
  - **Upper bound.** ex) $O(n^3) = n^2 $



#### 2. Big-Omega Notation

- 정의
  -  $T(n) = \Omega (f(n)) $ if and only if there exist constant $c, n_0$ such that $T(n) >= c f(n)$ for all $n >= n_0$
  - **Lower bound.** ex) $\Omega(n) = n^2$



#### 3. Big-Theta Notation

- 정의
  - $T(n) = \Theta(f(n))$ if and only if  $T(n) = \Omega (f(n)) $ and  $T(n) = \Omicron (f(n)) $
  - It is equivalent of the following: there exist constant $c_1, c_2, n_0$ such that $c_1f(n) <= T(n) <= c_2f(n)$ for all $n >= n_0$



#### 4. Little-O Notation

- 정의
  - $T(n) = o(f(n))$ if and only if **all contants $c >0$**, there exists a constant $n_0$ such that $T(n) <= c  f(n) $ for all $n >= n_0$
  - Big O Notation보다 엄격하다.
