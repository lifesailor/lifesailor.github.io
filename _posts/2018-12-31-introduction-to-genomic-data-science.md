---
title: "Introduction to genomic data science"
categories:
   - biology
tags:
   - biology
   - genomics
---

생물 분야의 Data Science에 관심이 생겨서 공부를 하고 있습니다. 고등학교 졸업 이후에 생물은 처음이어서 새로운 용어들에 익숙해지려고 노력하고 있습니다. 오늘은 Johns-hopkins 대학에서 진행하는 [Introduction to Genomic Technologies](https://www.coursera.org/learn/introduction-genomics) 라는 수업이 있어서 해당 수업을 정리해보고자 합니다.

<br/>

Introduction to Genomic Techonologies 수업은 크게 4가지 파트로 구성됩니다.

- Overview - Genomics, Molecular Biology, Human Genome Project에 대해서 간략한 소개를 합니다.
- Measurement Techonology - Genomics에 사용되는 측정 방법을 소개합니다.
- Computing Techonology - 기본적인 Computing Technology를 소개합니다.
- Data Science Techonology - 기본적인 통계 내용을 소개합니다.

<br/>

이 글에서는 위 4가지 파트 중에서 Overview와 Measurement Technology에 대해서 정리해보겠습니다. 만약 Gene, Genomics와 같은 용어에 익숙하지 않으신 분들은 [Genomics 101](https://lifesailor.github.io/genomics/genomics-1/)을 정리한 글을 보시면 도움이 될 것이라 생각합니다.

<br/><br/>

## Overview

- What is genomics?

  - 유전체의 구조, 기능, 진화와 관련된 분자 생물학의 분과. (The branch of molecular biology concerned with the structure, function, evolution and mapping of genomes)

  - 유전체의 연구와 관련된 분자 유전학의 분과, 특히 유전자의 시퀀스를 밝히고 그것을 바탕으로 의학, 약학, 농업 등에 응용. (The branch of molecular genetics concerned with study of genomes, specifically the identification of sequencing of their constituent genes and application of this knowledge in medicine, pharmacy, argriculture.)

<br/>

- genetics과 genomics의 차이
  - 유전학은 하나 또는 적은 수의 유전자를 다루지만 유전체학은 유전차 전체를 다룬다.

<br/>

- Genomic Data Science

![](/assets/images/biology/genomic-data-science.png)

<br/>

- Genomic Data Science 절차
  - 데이터 생성
  - [Reference Genome](https://ko.wikipedia.org/wiki/%ED%91%9C%EC%A4%80_%EA%B2%8C%EB%86%88) 과 서열 비교
    - 전처리 필요
  - Statistics & Machine Learning 적용

<br/>

- Human Genome project
  - 인간의 유전체에 대해서 Sequencing(3억 * base pairs)
  - 2001년에 완료되었다.
  - 현재 3,000,000 base마다 1달러 이하이다. 이전보다 저렴해졌다.

<br/>

## Measurement Technology

- PCR(Polymerase Chain Reaction)

  - 원하는 DNA 부분을 복제하여 증폭시키는 기술이다. 

  - PCR 필수 요소
    - 복제할 DNA
    - Primer
    - DNA 중합효소
    - DNA 중합요소 용 완충 용액

  - 과정
    - Denaturation: 이중가닥 DNA를 단일가닥 DNA로 분리한다.
    - Annealing: Primer가 결합한다.
    - Elongation: DNA를 복제한다.

 ![](/assets/images/biology/pcr.png)

<br/>

- NGS(Next Generation Sequencing)
  - 1세대(Sanger)/2세대 - PCR을 이용 
  - 3세대: PCR Amplification 운용 없이 DNA 분자 합성을 기반으로 High Fluorescend Detection 시스템을 이용한다.

<br/>

 ![](/assets/images/biology/sequencing.png)

<br/>

- Sequencing의 응용
  - 기본적인 아이디어: DNA로 변경한 다음에 Sequencing을 적용한다.
  - Exon Sequencing, RNA Sequencing, Chip-Sequencing











