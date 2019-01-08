---
title: "Data Structure and Algorithm(2) - Abstract Data Type"
categories:
  - data structure and algorithm
tags:
  - data structure
  - algorithm
---

[Princeton](https://www.coursera.org/learn/algorithms-part1) 대학의 로버트 세지윅 교수의 알고리즘 책을 보면서 자료구조와 알고리즘을 정리하고 있습니다. 해당 책에서는 java 언어를 사용합니다. 공부한 내용을 간단한 설명과 예제 코드로 정리하겠습니다. 단, 해당 글에서 정리한 자바 코드를 실행하기 위해서는 다음 [링크](https://algs4.cs.princeton.edu/code/)에 있는 Standard I/O 라이브러리를 사용할 자바 코드와 같은 경로에 다운 받아야 합니다.

<br/>

## 1. Abstract data type

- 추상 데이터 타입을 구현하는 것은 static 메서드를 통해 라이브러리를 만드는 것과 비슷하다. java에서 추상 데이터 타입은 자바 클래스로써 만들어지고 클래스 이름과 같은 이름을 가지는 .java 파일에 코드를 구현해 넣게 된다.

<br/>

## 2.  API, 클라이언트, 구현

- java에서 추상 데이터 타입을 만들 때는 API, 클라이언트, 구현 세 가지를 고려해야 한다.
- 이 책에서는 데이터 타입의 구현부와 그 클라이언트를 서로 독립시켜야 한다는 점을 강조하기 위해서 클라이언트 코드를 별도의 클래스로 만들어 main() 메서드를 만든다.
- 데이터 타입 클래스에 속한 main() 메서드에는 최소한의 단위 테스트를 위한 것만 남겨둔다.

<br/>

## 3. API 구현

데이터 타입을 구현할 떄 같은 단계를 밟는다.

- API 정하기
  - API의 목적은 클라이언트 코드와 데이터 타입 코드를 서로 독립적으로 만들어 모듈러 프로그래밍이 가능하게 하는 데 있다. API를 정할 때 두 가지 목표가 있다.
  - 첫 번째 클라이언트 코드를 명확하고 올바르게 하는 것이다. API 목록을 확정하기 전에미리 클라이언트 코드를 작성해보고 데이터 타입의 연산에 어떤 것들이 있는지 미리 부딪혀 보는 게 바람직하다. 이런 과정을 거치면 API 목록이 올바르게 정해지는지 좀 더 확신을 가질 수 있다. 두 번째는 구현 가능한 API를 정의하는 것이다. 어떻게 구현해야 하는 지 알지 못하는 API를 정의하는 것은 아무런 의미가 없다.
- 정해진 API에 맞게 자바 클래스를 구현한다. 먼저 인스턴스 변수를 선택하고 그 다음에 생성자와 인스턴스 메서드를 구현한다.
- 복수의 테스트 클라이언트를 만든다.이 테스트 클라이언트들은 앞서의 두 과정에서 선택한 설계상의 선택들이 올바르게 동작하는 검증한다.

<br/>

##4. 추상 데이터 타입 설계

추상 데이터 타입은 내부 데이터 표현방식을 외부 클라이언트로부터 숨긴다. 객체 지향 프로그래밍의 가장 대표적인 특징은 데이터 타입의 구현부를 숨김으로써 클라이언트 코드 개발과 데이터 타입 구현을 서로 독립적으로 할 수 있게 하는 것이다. 은닉화는 모듈러 프로그래밍을 가능하게 하는 열쇠로 다음과 같은 것들을 가능하게 해준다.

- 클라이언트 개발과 데이터 타입 구현을 분리시킨다.
- 클라이언트 코드에 영향을 주지 않고 데이터 타입 구현의 개선사항을 적용할 수 있다.
- 아직 작성되지 않는 프로그램을 지원할 수 있다.

은닉화는 데이터 타입 연산도 격리시켜준다. 이를 통해 아래와 같은 것들이 가능해진다.

- 잠재적 오류 가능성을 제한시킨다.
- 구현부에 일관된 검증과 디버깅 기능을 넣을 수 있다.
- 클라이언트 코드를 명료하게 할 수 있다.

<br/>

## 5. 1.2까지 다룬 java 클래스 요약

- Static method: 인스턴스 변수가 없다.
  - 예시: Math, StdIn, StdOut

<br/>

- 불변 추상 데이터 타입: 모든 인스턴스 변수가 private이며 final로 선언되어 참조 타입으로 복제되는 것에 대응한다.
  - 예시: Date, Transaction, String, Integer

```java
// Date.java
public class Date {
    private final int month;
    private final int day;
    private final int year;

    public Date(int m, int d, int y)
    { month = m; day = d; year = y; }

    public int month()
    { return month; }

    public int day()
    { return day; }

    public int year()
    { return year; }

    public String toString()
    { return month() + "/" + day() + "/" + year(); }

    public static void main(String[] args)
    {
        int m = Integer.parseInt(args[0]);
        int d = Integer.parseInt(args[1]);
        int y = Integer.parseInt(args[2]);

        Date date = new Date(m, d, y);
        StdOut.println(date);
    }
}
```

<br/>

- 가변 추상 데이터 타입: 모든 인스턴스 변수가 private이다. 하지만 모든 인스턴스 변수가 final일 필요는 없다.
  - Counter, Accumulator

```java
// Counter.java
public class Counter {

    // 인스턴스 변수
    private final String name;
    private int count;

    public Counter(String id)
    { name = id; }

    public void increment()
    { count++; }

    public int tally()
    { return count; }

    public String toString()
    { return count + " " + name; }

    public static void main(String[] args)
    {
        Counter heads = new Counter("heads");
        Counter tails = new Counter("tails");

        heads.increment();
        heads.increment();
        tails.increment();

        // 자동으로 toString 호출
        StdOut.println(heads + " " + tails);

        // 메서드 호출
        StdOut.println(heads.tally() - tails.tally());
    }
}
```

- I/O 부가 효과가 있는 추상 데이터 타입: 모든 인스턴스 변수가 private이다. 인스턴스 메서드에서 I/O가 발생한다.
  - VisualAccumulator, In, Out, Draw

```java
// VisualAccumulator.java
public class VisualAccumulator {
    private double total;
    private int N;

    public VisualAccumulator(int trials, double max)
    {
        StdDraw.setXscale(0, trials);
        StdDraw.setYscale(0, max);
        StdDraw.setPenRadius(.005);
    }

    public void addDataValue(double val)
    {
        N++;
        total += val;
        StdDraw.setPenColor(StdDraw.DARK_GRAY);
        StdDraw.point(N, val);
        StdDraw.setPenColor(StdDraw.RED);
        StdDraw.point(N, mean());
    }

    public double mean()
    { return total / N; }

    public String toString()
    { return "Mean (" + N + " values): " + String.format("%7,5f", mean()); }

    public static void main(String[] args) {
        int T = Integer.parseInt(args[0]);
        VisualAccumulator a = new VisualAccumulator(T, 1.0);
        for (int t = 0; t < T; t++)
            a.addDataValue(StdRandom.random());
        StdOut.println(a);
    }
}
```