---
title: "Data Structure and Algorithm(3) - Bag, Queue, Stack"
categories:
  - data structure and algorithm
tags:
  - data structure
  - algorithm

---

[Princeton](https://www.coursera.org/learn/algorithms-part1) 대학의 로버트 세지윅 교수의 알고리즘 책을 보면서 자료구조와 알고리즘을 정리하고 있습니다. 해당 책에서는 java 언어를 사용합니다. 공부한 내용을 간단한 설명과 예제 코드로 정리하겠습니다. 

<br/>

## 1. API

- Bag: 항목을 삭제할 수 없는 컬렉션이다. Bag의 목적은 항목을 수집하고 수집한 항목을 순회할 수 있는 도구를 제공한다.
  - Bag(): 공백 Bag 생성
  - void add(Item item): 항목 추가
  - boolean isEmpty(): 백이 비어있는가.
  - int size(): 백에 든 항목의 개수

<br/>

- Queue(FIFO): 선입선출 큐는 먼저 들어간 것이 먼저 나오는 것을 정책으로 하는 컬렉션이다.
  - Queue(): 공백 큐 생성
  - void enqueue(Item item): 항목 추가
  - Item dequeue(): 최근에 추가된 항목 제거
  - boolean isEmpty(): 큐가 비어있는가.
  - int size(): 큐에 든 항목의 개수

<br/>

- Stack(LIFO): 후입선출 스택은 마지막으로 들어간 것이 가장 먼저 나오는 정책을 가지는 컬렉션이다.
  - Stack(): 비어 있는 스택 생성
  - void push(Item item): 항목 추가
  - Item pop(): 가장 최근에 추가된 항목 제거
  - boolean isEmpty(): 스택이 비었는가?
  - int size(): 스택에 든 항목의 개수

<br/>

## 2. Generic, Autoboxing, Iterator

- Generic
  -  컬렉션 ADT의 가장 핵심적인 특징은 어떤 타입의 데이터느 사용할 수 있어야 한다는 점이다. 자바의 제너릭 매커니즘은 이것을 가능하게 한다. 제너릭을 다른 말로 파라미터화된 타입이라고도 한다. 클래스 뒤에 기입된 <Item> 이란 표기는 API 구현부에서 사용된 Item 타입을 대체할 실제 타입을 의미한다. Stack <Item> 은 아이템들의 스택이라고 읽으면 이해하기 쉽다.

```java
Stack <String> stack = new Stack <String>();
stack.push(new Stack(12, 31, 1991));

Stack <Date> stack = new Stack <Date>();
stack.push(new Date(12, 31, 1991));
```

<br/>

- Autoboxing
  -  제네릭 타입 파라미터로부터 생성되는 변수는 참조형 타입으로만 생성되어야 한다. 이 떄문에 int와 같은 기본 데이터 타입이 문제가 되는데 자바에서는 기본 데이터 타입도 제네릭 코드에서 활용할 수 있도록 특별한 매커니즘을 지원한다. Boolean, byte, char, double, float, int, long, short 각각에 대응하여 Byte, Character, Double, Float, Integer, Long, Short가 존재한다. 자바는 기본 데이터 타입과 상응되는 참조형 타입을 자동으로 변환해준다.

```java
Stack <Integer> stack = new Stack <Integer>();
stack.push(17); // auto-boxing
int i = stack.pop(); // auto-unboxing
```

<br/>

- Iterator
  - 많은 수의 어플리케이션들이 컬렉션 항목들 각각에 대해 어떤 처리를 하거나 컬렉션의 모든 항목을 순회할 수 있는 기능을 필요로 한다. 컬렉션 항목에 대한 순회 개념은 자바를 포함한 현대 프로그래밍 언어들이 최우선순위로 달성하고자 하는 패러다임이다. 단지 라이브러리가 아니라 언어 차원에서 그러한 매커니즘을 지원한다. 이 매커니즘을 반복자라 부른다. 반복자를 통해 컬렉션의 세부적인 구현 방식에 신경쓰지 않고서도 명료하고 압축적인 코드를 작성할 수 있다.
  - 이러한 (1) foreach 구문은 (2) while 구문의 축약형과도 같다. 즉 아래와 같은 while 구문과 동일하다 

```java
/*
(1) foreach
*/

Queue <Transaction> collection = new Queue <Transaction>();

for (Transaction t: collection)
{	StdOut.println(t);	}
```

```java
/*
(2) while
*/
Iterator <String> i = collection.iterator();
while (i.hasNext())
{
	String s = i.next();
    StdOut.println(s);
}
```

- 위 코드는 컬렉션이 반복자를 구현하는 데 필요한 요소들을 드러낸다. 
  - 컬렉션은 Iterator 객체를 리턴하는 iterator() 메서드를 구현해야 한다.
  - Iterator 클래스는 두 개의 메서드 hasNext()(boolean 리턴 값)와 next()(컬렉션의 제너릭 항목을 리턴한다)를 구현해야 한다.
- 자바에서는 어떤 특정 클래스가 특정 메서드를 지원함을 표시하기 위해 인터페이스라는 매커니즘을 지원한다. 컬렉션이 반복자를 지원함을 나타낼 수 있도록 이미 자바 자체적으로 필수 인터페이스를 정의하고 있다. 어떤 클래스가 반복자를 지원하기 위해서는 먼저 클래스 선언부에 implements Iterable <Item> 구문을 추가하여 Iterable 인터페이스를 따른다는 것을 표현해야 한다.

```java
// Iterable - 반복자
public interface Iterable <Item>
{
	Iterator<Item> iterator();
}

// 스택에서 저장한 배열을 역순으로 순회해야 한다.
public Iterator<Item> iterator()
{
    return new ReverseArrayIterator();
}

// Interator 
public interface Iterator <Item>
{
	boolean hasNext();
	Item next();
	void remove();
}

// Iterator
private class ReverseArrayIterator implements Iterator <Item>
{
	private int i = N;
	
	public boolean hasNext(){	return i > 0;	}
	public Item next(){	return a[--i];	}
	public void remove(){	}
}
```

