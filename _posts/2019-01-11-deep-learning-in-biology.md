---
title: "Opportunities and obstacles for deep learning in biology and medicine"
categories:
  - biology and medicine
tags:
  - paper
  - biology
  - review
---

[Opportunities and obstacles for deep learning in biology and medicine, 2018](https://www.biorxiv.org/content/early/2017/05/28/142760)

<br/>

해당 논문은 생물 및 의학 분야에서  deep learning 어플리케이션을 다루었습니다. 서른 명이 넘는 저자가 같이 작성하고 123쪽에 이를 정도로 방대한 내용을 담고 있습니다. 저는 이 논문 전체를 소개하는 초록 및 서론 일부분과 제가 현재 관심 있게 살펴 보는 몇 가지 부분을 중심으로 번역해보겠습니다. 해당 논문의 전체 목차는 다음과 같습니다. 해당 목차에서 제가 번역할 부분에 대해서는 **볼드**처리를 해두었습니다.

<br/>

## 목차

- **Abstract(초록) $\tag{1}$** 

- Introduction to deep learning(딥러닝 소개)

  - **Will deep learning transform the study of human disease?(딥러닝이 의학 연구를 변화시킬 것인가?) $\tag{2}$**

    - Disease and patient categorization(질병과 환자 분류에 적용)
    - Fundamental of biological study(생물의 기본 분야에 적용)
    - Treatment of patients(환자 치료 분야에 적용)

    

  - **Deep learning and patient categorization(딥러닝과 환자 분류) $\tag{3}$**

    - Imaging applications in healthcare(헬스 케어 분야에서 이미지 어플리케이션)
    - Text applications in healthcare(헬스 케어 분야에서 텍스트 어플리케이션)
    - Electronic health records(전자 건강 기록)
    - Challenges and opportunities in patient categorization(환자 분류 분야에서 어려움과 기회)

- Deep learning to study the fundamental biological processes underlying human disease(인간 질병을 일으키는 근본적인 생물 연구에서 딥러닝)
  - Gene expression(유전자 표현형)
  - Splicing(스플라이싱)
  - Transcription factors(전사 인자)
  - Promoters and enhancers(프로모터와 인핸서)
  - Micro-RNA binding(마이크로 RNA 바인딩)
  - Protein secondary and tertiary structure(단백질 2,3차 구조)
  - Structure determination and cry-electron microscopy(구조 결정 및 전자 현미경)
  - Protein-protein interactions(단백질간 상호작용)
  - MHC-pesticide binding(MHC-pesticide binding)
  - PPI networks and graph analysis(PPI 네트워크 및 그래프 분석)
  - Morphological phenotypes(형태 표현형)
  - Single-cell data(단세포 데이터)
  - Metagenomics(메타 유전체)
  - Sequencing and variant calling(시퀀심 및 변종 호출)
  - Neuroscience(신경과학) $\tag{4}$
- The impact of deep learning in treating disease and developing new treatment
  - Clinical decision making(임상 의사 결정)
  - Drug repositioning(약물 repositioning)
  - Drug development(약물 개발)
- Discussion(토의)

- Conclusion(결론)
- Methods(논문 작성 방법)
- Reference(레퍼런스) 

<br/>

## 1. Abstract(초록)

머신러닝의 한 종류인 딥러닝은 최근에 다양한 영역에서 인상적인 결괄르 보여주고 있다. 생물과 의학 분야는 데이터가 많지만 데이터가 복잡하고 이해하기 어렵다. 이런 성질은 딥러닝에 적합하다. 우리는 딥러닝을 환자 분류, 근본적인 생물 과정, 환자 치료 분야 등 다양한 생의학적인 문제에 적용하는 것을 검토하고 딥러닝이 이러한 과제를 변화시킬지 아니면 생의학계의 고유한 영역의 문제가 있을 지 논의한다. 우리는 딥러닝이 아직 이 문제들 중 어떤 것도 혁신하지는 못했지만, 이전보다 유망한 진보가 이루어졌다는 것을 발견했다. 이전과 비교했을 때 개선이 미미한 경우에도 딥러닝이 사람이 연구를 빠르게 하거나 도울 수 있다는 것을 봤다. 해석 가능성과 각 문제를 가장 잘 모델링하는 방법을 찾기 위해 더 많은 작업이 필요하다. 게다가 제한된 양의 트레이닝 데이터 및 법적 프라이버시 제약 등은 일부 영역에서 문제를 제기한다. 그럼에도 불구하고, 우리는 생물 및 의학 분야에서 딥러닝이 변화를  가져올 것이라고 예상한다.

<br/>

## 2. 딥러닝이 질병 연구를 혁신할 수 있을까?

이 논문에서 우리는 사람들이 건강을 유지하고 회복하기 위해 분류하고 연구하고 치료하는 방법을 혁신하기 위해서 딥러닝에게 필요한 것이 무엇인지 질문합니다. 우리는 혁신이라는 단어에 높은 기준을 설정합니다. 인텔의 전 CEO인 Andrew Grove는 전략적 변곡점이라는 단어를 만들었다. 이 단어는 기업의 비즈니스가 근본적으로 재구조화하도록 하는 기술과 환경의 변화를 일컫는다. 여기에서 우리는 딥러닝이 생물학과 의학의 실무에서 전략적 변곡점을 유도할 수 있는 혁신인지 확인하고자 한다.

이미 생물학에서 딥러닝, 의료 및 약물 개발 적용에 초점을 둔 연구들이 있었다. 우리는 연구자들이 이전에 불가능하다고 여겼거나 지루하게 분석한 문제를 해결한 사례를 찾았다. 우리는 또한 연구자들이 생물 의학적 데이터에 의해 제기된 도전을 회피하기 위해 사용하고 있는 접근법을 확인했다. 우리는 도메인적인 요소를 고려하는 것이 딥러닝을 이용하는데 중요하다는 것을 발견했다. 모델 해석도 또한 중요하다. 데이터의 패턴을 이해하는 것은 데이터를 적합시키는 것만큼이나 중요하다. 게다가 데이터의 기본 구조를 효율적으로 대표하는 네트워크를 구축하는 방법에 관한 중요한 질문이 있다. 도메인 전문가는 데이터를 정확하게 표현하도록 네트워크를 디자인하고 사전 지식을 인코딩하고 성공 또는 실패를 평가할 때 중요한 역할을 한다. 또한 실험의 우선순위를 정하거나 전문가의 판단이 필요하지 않은 작업을 간소화하여 생물학자 및 임상의사를 강화하는 것도 큰 잠재력이 있다. 우리는 광범위한 주제를 질병 및 환자 분류, 근본적인 생물 연구, 환자의 치료 3개로 나누었다. 여기에서 우리는 간략하게 각각의 종류의 질문을 소개하고 각각의 유형의 문제에 대한 접근 방법이나 데이터에 대해서 소개한다.

<br/>

### 질병과 환자 분류

생체 의학의 주요 과제는 질병과 질병 하위 유형의 정확한 분류다. 종양학에서, 현재의 "기준" 접근법은 전문가들의 해석이나 세포 표면 수용체나 유전자 표현과 같은 분자 표지의 평가를 필요로 하는 역사학을 포함한다. 한 가지 예는 유방암을 분류하기 위한 PAM50 접근법으로, 유방암 환자를 네 가지 하위 유형으로 분류하는 것이다. 이 네 가지 하위 유형에는 여전히 실질적 이질성이 존재한다. 이용 가능한 분자 데이터의 부피가 증가하는 것을 고려하면, 보다 포괄적인 하위 유형 지정이 가능할 것으로 보인다. 몇 가지 연구는 유방암 환자를 더 잘 분류하기 위해 딥러닝을 사용했다. 예를 들어, 관리되지 않는 접근방식인 자동 인식기를 제거하는 것은 유방암 환자를 군집화하는 데 사용될 수 있으며 CNN은 암 결과와 높은 상관관계를 가지는 특징인 유사 분열을 셀 수 있다.

<br/>

### 근본적인 생물학 연구

딥러닝은 근본적인 생물학 질문에 대합하기 위해 사용될 수 있다. 특히 고처리량 '체학'연구에서 대량의 데이터를 처리하는 데 적합하다.  머신러닝과 딥러닝이 광범위하게 적용되는 분야는 분자 타겟 예측이다. 예를 들어서 순환 신경망은 microRNA 유전자 타겟을 예측하는데 사용될 수 있다. 그리고 CNN은 단백 residue - residue 상호작용과 2차 구조를 예측하기 위해서 적용되었다. enhancer와 promoter와 같은 기능적 게놈 요소의 인식과 뉴클레오타이드 다형성의 해로운 영향에 대한 예측 또한 최근에 딥러닝이 적용되는 사례다.

<br/>

### 환자 치료

환자 치료에 대한 딥러닝의 적용이 이제 막 시작되었지만, 우리는 딥러닝이 환자 치료를 추천하고, 치료 결과를 예측하고 새로운 치료법을 안내하도록 추천할 것이라 기대한다. 이 분야에서 한 가지는 약물 타겟을 확인하고 약물 반응을 예측하는 것을 목표로한다. 또 다른 연구에서는 약물 상호 작용과 약물 생체 활성을 예측하기 위해 단백질 구조에 대해서 딥러닝을 사용한다. transcriptomic 데이터에 딥러닝을 사용하는 신약 재창출 또한 흥미로운 영역이다. 제한 볼츠만 머신은 신약 - 타겟 상호작용을 예측하고 약물 재배치 가설을 공식화한다. 마지막으로 딥러닝은 새로운 타겟에 대한 약물 발견의 초기 단계에서 화학 물질을 우선순위를 정할 수 있다.

<br/>

## 3. 질병과 환자분류

개인은 증상, 특정 진단 테스트의 결과 또는 기타 요인에 따라 질병이나 상태를 진단 받는다. 일단 질병을 진단받으면, 개인에게는 인간이 정의한 다른 규칙에 따라 단계가 배정될 수 있다. 이러한 규칙은 시간이 지남에 따라 개선되지만, 그 과정은 진화적이고 임시적이며, 잠재적으로 기본적인 생물학적 메커니즘과 그에 상응하는 치료 개입의 식별을 방해한다.

환자 표현형의 대규모 말뭉치에 적용되는 딥러닝은 환자 분류에 대해 의미 있고 데이터 중심적인 접근방식을 제공할 수 있다. 예를 들어, 그들은 질병의 역사적 정의로 인해 가려질 수 있는 새로운 공유 메커니즘을 식별할 수 있다. 아마도 우리의 추정의 맥락 없이 데이터를 재평가함으로써, 깊은 신경망은 치료 가능한 상태의 새로운 종류를 밝힐 수 있을 것이다.

이러한 낙관론에도 불구하고 딥러닝 모델로 무차별적으로 예측 신호를 만들어내는 능력은 평가되고 신중하게 운영되어야 한다. 전자 건강 기록에서 얻은 임상 테스트 결과를 제공하는 깊은 신경망을 상상해 보십시오. 의사는 의심되는 진단을 기반으로 특정 테스트를 할 수 있기 때문에, 딥러닝 모델은 테스트 결과를 기반으로 환자를 진단하는 방법을 배울 수 있다. ICD (International Classification of Diseases) 코드를 예측하는 것과 같은 일부 객관적인 기능의 경우 의사 활동을 초월한 기본 질병에 대한 통찰력을 제공하지는 못하더라도 우수한 성능을 제공 할 수 있습니다. 이 과제는 딥러닝에게만 국한되지는 않는다. 하지만 실무자들이 이러한 당면 과제와 고도로 예측 가능한 분류자를 구성할 수 있는 이 영역의 가능성을 아는 것이 중요하다.

이 절에서 우리의 목표는 딥러닝이 이미 새로운 범주의 발견에 기여하고 있는 정도를 평가하는 것이다. 그렇지 않다면, 우리는 이러한 목표를 달성하기 위한 장벽에 초점을 맞춘다. 우리는 또한 연구자들이 특히 데이터 가용성과 라벨링과 관련하여 현장 내 문제를 해결하기 위해 취하고 있는 접근법을 강조한다.

<br/>

### 헬스케어 분야에서 이미지 어플리케이션

딥러닝 방법은 자연적 이미지와 비디오의 분석을 변화시켰고, 이와 유사한 사례들이 의료 영상과 함께 나타나기 시작했다. 병변과 결절을 분류하고, 장기, 지역, 랜드마크와 병변을 국지화하고, 장기, 기관 하부 구조 및 병변을 지역화하고, 내용을 기반으로 이미지를 검색하고, 이미지를 생성 및 강화하고, 이미지를 임상 보고서와 결합하는 데 딥러닝이 사용되어 왔다.

의료 영상 분석은 자연 이미지 분석과 많은 공통점이 있지만 주요 차이점도 있다. 우리가 조사한 모든 사례에서, 훈련에 사용할 수 있는 이미지가 1백만 개 미만이었으며, 데이터 집합은 자연 이미지 모음보다 훨씬 작은 경우가 많다. 연구원들은 이 과제를 해결하기 위해 하위 작업별 전략을 개발하였다.

데이터 증강은 작은 훈련 세트를 사용하기 위한 효과적인 전략을 제공한다. 이러한 관행은 유방조영술의 영상을 분석하는 일련의 논문에서 나타난다. 이미지 수와 다양성을 확대하기 위해 연구원들은 적대적 훈련 사례를 구축했다. 적대적 훈련 사례들은 훈련 이미지는 내용은 변경하지 않지만 약간의 변형을 주는 방식으로 만들어진다. (예: 이미지를 임의의 양으로 회전함)을 적용하여 작성된다. 의료 영역에서 대안은 미세 조정에 앞서 인간이 만든 특징을 향해 훈련하는 것이다. 이는 특징을 추출하는 딥러닝 기법의 역량을 포기하더라도 이 도전을 회피하는 데 도움이 될 수 있다.

두 번째 전략은 ImageNet과 같은 딥러닝 모델에 의해 자연 이미지에서 추출된 특징을 새로운 목적으로 재활용하는 것이다. 2015년 Kaggle 대회에서 레이블이 있는 대형 이미지 세트를 공개적으로 사용할 수 있게 한 후 컬러 안저 이미지를 통한 당뇨병 망막증 진단은 딥러닝 연구자들에게 초점을 맞춘 영역이 되었다. 대부분의 참가자는 신경망을 처음부터 훈련시켰으나 Gulshan 등은 이를 훈련시켰다. 자연 이미지로 사전 훈련된 48단 Inception-v3 architecture를 사용하여 가장 좋은 specifiity와 sensitivity를 얻었다. 그러한 특징들은 또한 피부암의 가장 치명적인 형태인 흑색종과 피부 병변의 비염색 이미지 및 연령과 관련된 시력 감퇴를 감지하기 위해서도 사용된다. 자연 이미지에 대한 사전 훈련은 매우 깊은 네트워크들이 과적하지 않고 성공할 수 있게 한다. 악성흑생종의 경우, 보고된 성능은 인증된 피부과 의사 위원회와 비교하거나 더 우수했다. 방사선 이미지에 대해서도 feature를 재사용하는 것이 새로운 접근 방법으로 떠오르고 있다. 자연 이미지에 대해 훈련된 CNN은 방사선 이미지에서 성능을 향상시킨다. (중략)

다른 작업에서 특징을 추출해서 사용하는 기술은 transfer learning 입니다. 비록 우리가 새 작업에 자연 이미지 특징을 전환한 성공 사례들을 언급했지만, 몇몇은 부정적인 결과가 있었을 것이라 추정합니다. MRI 이미지도 작은 훈련 세트의 문제에 직면해있다. 이 도메인에서 Amit은 사전 학습된 모델과 MRI 이미지 만으로 훈련한 이미지의 tradeoff를 조사했다. 다른 문헌과 달리 그들은 수십 명의 환자로부터 얻은 수백 개의 이미지에 대한 데이터 증강 방법을 활용해서 학습한 작은 네트워크가 도메인이 다른 사전 학습된 분류기보다 좋은 성능을 낸다는 것을 발견했다.

제한된 훈련 데이터를 다룰 수 있는 방법은 3D 이미지를 projection을 통해서 수많은 이미지로 나누는 것이다. Shin은 컴퓨터 단층 촬영 이상 감지를 위한 다양한 심층 네트워크 아키텍처, 데이터 세트 특성 및 교육 절차를 비교했다. 그들은 훈련 데이터 집합의 제한된 크기에도 불구하고 22층까지의 네트워크가 3D 데이터에 유용할 수 있다고 결론지었다. 그러나, 그들은 아키텍처, 매개변수 설정 및 모델 미세 조정의 선택이 매우 문제점과 데이터 집합마다 다르다는 점에 주목하였다. 더욱이, 이러한 유형의 작업은 종종 병변 국소화와 외모 모두에 달려있으며, 이는 CNN 기반 접근에 어려움을 제기한다. 표준 신경 네트워크 아키텍처를 통해 3차원의 전체 이미지에서 유용한 정보를 동시에 캡처하려는 직선적 시도는 계산적으로 불가능했다. 대신에, 2차원 모델을 사용하여 영상 슬라이스를 개별적으로 처리하거나(2D) 네이티브 공간(2.5D)의 여러 2D 투영에서 정보를 수집했다.

Roh는 CT 스캔으로부터 2D, 2.5D, 3D CNN을 많은 detection 작업에 대해서 비교했고, 2.5D CNN이 3D CNN보다 시간도 적게 걸릴 뿐 아니라 성능도 좋았다. 특히 훈련 데이터를 증강했을 때 그러했다. 또 다른 2D와 2.5D 네트워크의 이점은 사전 학습된 모델을 사용할 수 있다는 점이다. 그러나 차원을 축소하는 것은 도움이 되지 않았다. Nie 는 다중 모드, 다중 채널 3D 심층 아키텍처가 MRI, 기능적 MRI 및 확산 MRI 이미지에서 높은 수준의 뇌종양 출현 특징을 학습하는 데 성공하여 단일 모달 또는 2D 모델보다 우수한 성능을 보임을 보여주었습니다. 전반적으로, 훈련 세트의 다양성, 속성 및 크기, 입력의 차원 및 의료 영상 분석에서의 최종 목표의 중요성은 자연 이미지와 다른 전문화된 딥러닝 네트워크의 개발, 훈련 및 검증 방법, 입력 표현을 요구한다.

딥러닝 모델로부터 예측한 것은 전문가에 의해서 평가될 수 있다. 많은 유방 조영술 이미지에서, Kooi는 딥러닝 모델이 다른 전통적인 컴퓨터 진단 시스템보다 낮은 감도에서 성능이 우수하고 높은 감도에서 비교할정도로 작동하는 것을 확인했다. 또한 인증된 방사선 학자가 네트워크 성능을 패치 수준에서 비교한 결과 네트워크와 사람간의 큰 차이가 없음으르 발견했다. 하지만 딥러닝 모델의 임상에서 문제는 각각의 예측마다 신뢰도를 부여하기 어렵다는 것이다. Leibig et al.는 dropout 네트워크를 베이즈 추론과 연결하여 당뇨 망막 병증 진단을위한 심층 네트워크의 불확실성을 예측했다. 각 예측에 신뢰도를 부여하는 기술은 병리학자와  컴퓨터간의 상호 작용을 돕고 의사의 이해를 향상시켜야합니다.

딥러닝은 조직 슬라이드 분석을 돕는 시스템으로 유망하다. Ciresan은 조직 슬라이드 분석의 가장 초기 접근 법을 개발했고, 2012년 감수분열 Detection 경진대회에서 인간 수준의 정확도를 얻으며 승리했다. 보다 최근의 연구에서, Wang은 암을 확인하기 위해 림프절 절편의 염색 슬라이드를 분석했다. 이 작업에서 병리학자는 약 3 %의 오류율을 보였다. 병리학자는 오검을 일으키지 않았지만 과검을 했습니다. 딥러닝은 병리학자보다 약 2 배의 오류율을 냈지만 오류가 크게 상관을 가지지 않았습니다. 이 영역에서 이러한 알고리즘은 병리학자를 돕고 오탐률을 줄이기 위해 기존 도구에 통합 될 준비가되어있을 수 있습니다. 딥러닝과 사람의 조합은 데이터 제한에 의해 제시된 몇 가지 어려움을 극복하는 데 도움이 될 수 있습니다.

풍부한 표현형 주석이 포함된 학습 사례의 한 가지 소스는 EHR입니다. ICD 코드 형식의 청구 정보는 간단한 주석이지만 표현형 알고리즘은 실험실 테스트, 약물 처방 및 환자 노트를 결합하여보다 신뢰할 수있는 표현형을 생성 할 수 있습니다. 최근, Lee는 연령 관련 황반변 성 환자를 대조군과 구별하는 접근법을 개발했다. 구조화 된 전자 건강 기록에서 추출한 약 10 만 개의 이미지에서 심층 신경 네트워크를 교육하여 93% 이상의 정확도를 달성했습니다. 저자들은 훈련을 멈출 시기를 평가하기 위해 테스트 세트를 사용했습니다. 다른 영역에서도 예상 정확도가 거의 변하지 않았지만, 가능하면 독립적인 테스트 세트를 구성해야 합니다.

풍부한 임상 정보는 EHR에 저장됩니다. 그러나 대형 세트에 수동으로 주석을 달려면 전문가가 필요하며 시간이 많이 걸립니다. 흉부 X 선 검사의 경우, 방사선과 의사는 대개 예제마다 몇 분을 소비합니다. 딥러닝에 필요한 예제를 생성하는 것은 비용이 많이 들지 않습니다. 그 대신, 연구자는 텍스트 주석을 사용하여 주석을 생성하는 이점을 누릴 수 있습니다. 주석이 완전히 정확하지 않다고 할 지라도. Wang은 조금의 라벨이 있는 이미지를 사용하여 예측 신경 네트워크 모델을 구축 할 것을 제안했다. 이러한 레이블은 자동으로 생성되며 사람이 확인할 수 없기 때문에 시끄럽거나 불완전 할 수 있습니다. 이 경우, 관련 흉부 X 선 방사선 보고서에 일련의 자연 언어 처리 (NLP) 기술을 적용했습니다. 그들은 최첨단 NLP 도구를 사용하여 보고서에 언급 된 모든 질병을 먼저 추출한 다음 NegBio라는 새로운 방법을 적용하여 보고서에서 부정확하고 모호한 결과를 필터링했습니다. 4 개의 독립적인 데이터 세트에 대한 평가는 NegBio가 부정확하고 모호한 결과를 발견하는 데 매우 정확하다는 것을 보여주었습니다. 결과 데이터 세트는 30,805 명의 환자로부터의 112,120 개의 정면 - 흉부 엑스선 영상으로 구성되었으며, 각 이미지는 (약 표식) 병리학 카테고리, (예 : 폐렴 및 심 부정맥)와 관련되거나 아닌 것으로 textmining되었습니다. 또한, Wang은 일반적인 흉부 질환을 발견하기 위해 통합된 weakly supervised 다중 레이블 이미지 분류 체계를 사용했습니다. 이는 완전히 라벨링 된 데이터를 사용하여 벤치 마크보다 우수한 성능을 보였습니다.

자연 이미지와 같은 문제 (예 : 흑색 종 검출)를 제외하고는 생의학 이미지 분석은 심층적인 학습을 위해 여러 가지 과제를 제기합니다. 데이터 세트는 일반적으로 작고 주석은 희소 할 수 있으며 이미지는 종종 고차원, 다중 모달 및 다중 채널입니다. transfer learning,  데이터 Augmentation 및 multi-view 및 multi-stream 아키텍처 사용과 같은 기술이 더 일반적입니다. 또한, 높은 모델 민감성과 특이성은 임상적 가치로 직접 해석 될 수 있습니다. 따라서 예측 평가, 불확실성 추정, 모델 해석 방법 또한 이 영역에서 는 매우 중요합니다.. 마지막으로 딥러닝 방법의 힘을 인간의 전문 기술과 결합하여 환자 치료 및 관리에 대한 정보에 입각 한 의사 결정을 내릴 수있는 더 나은 병리학 의사 - 컴퓨터 상호 작용 기술이 필요합니다.

<br/>

### 헬스케어 분야에서 자연어 처리

학술 출판물과 EHR의 급속한 성장으로 인해 최근 몇 년 동안 생물 의학 텍스트 마이닝이 점점 중요 해지고 있습니다. 생물학 및 임상 텍스트 마이닝의 주요 작업에는 개체 인식, 관계 / 이벤트 추출 및 정보 검색이 포함됩니다. 전통적 방법과 피처 엔지니어링의 어려움을 극복 할 수 있고 성능이 좋기 떄문에 이 영역에서 깊은 학습이 매력적입니다. 응용 분야 (생체 의학 문헌 대 임상 노트)와 실제 작업(예 : 개념 또는 관계 추출)에 따라 관련 응용 프로그램을 계층화 할 수 있습니다. 

개체 인식은 제어된 어휘 또는 온톨로지에서 특정 클래스의 생물학적 개념을 참조하는 텍스트 범위를 식별하는 작업입니다. NER은 종종 복잡한 텍스트 마이닝의 첫 단계로 필요합니다. 최첨단 방법은 일반적으로 작업을 시퀀스 라벨링 문제로 구성하고 조건부 random field를 사용합니다. 최근에는 단어의 풍부한 잠재적 의미 정보를 포함하는 word embedding이 NER 성능을 높이는 데 널리 사용되었습니다. Liu는 약물 이름 인식에 word embedding과 기존 의미론적 특징을 비교했다. Tang은 gene, DNA, cell line에 word embedding 사용을 조사했습니다. 게다가, Wu는 임상 약어 명확화를 하기 위해 word embedding 사용을 검토했다. Liu는 임상 약어 확장을 위한 word embedding을 배우는 과제 중심적 자원을 개척했다.

(중략)

정보 검색은 대용량 문서 수집에서 정보 요구를 충족시키는 관련 텍스트를 찾는 작업입니다. 딥러닝이 아직 다른 분야에서 보인 것처럼이 분야에서 동일한 성공 수준을 달성하지 못했지만, 최근 관심과 노력의 급증은 이것이 빠르게 변화하고 있음을 시사합니다. 예를 들어, Mohan은 문헌의 텍스트를 질의에 연관시키는 딥러닝 접근 방식을 설명했다.이 접근 방식은 전체 생물 의학 문헌에 적용되었다.

요약하면, 심층 학습은 많은 생물 의학 텍스트 마이닝 작업 및 애플리케이션에서 유망한 결과를 보여줍니다. 그러나이 영역에서 그 잠재력을 최대한 발휘하기 위해서는 제한된 표지 된 데이터를 다루는 현재의 방법에서 큰 크기의 표식 된 데이터 또는 기술적 진보가 필요합니다.

<br/>

### 전자 건강 기록

EHR 데이터에는 상당한 양의 자유 텍스트가 포함되어 있으며  접근하기가 어렵습니다. 종종 특정 작업에서 잘 수행되는 알고리즘을 개발하는 연구원은 도메인 별 기능을 설계하고 구현해야합니다. 이러한 기능은 처리중인 문헌의 고유 한 측면을 포착합니다. 딥러닝 방법은 스스로 특징을 학습합니다. 최근 연구에서, 저자는 도메인 별 개념 추출을 위한 포괄적인 기능들에 적용될 수있는 딥러닝 학습 방법의 정도를 평가했다. 그들은 성과가 가장 좋은 도메인 특정 방법보다 낮지 만 성능이 떨어지는 것을 발견했습니다. 이는 특정 솔루션을 개발하는 데 필요한 연구원의 시간과 비용을 줄임으로써 딥러닝이 현장에 영향을 미칠 수있는 가능성을 제기하지만 성능 향상으로 이어지지는 않을 수도 있습니다.

<br/>

### 환자 분류 영역에서 도전과 기회

- **Generating ground-truth labels can be expensive or impossible**(정확한 레이블을 만드는 것이 불가능하거나 비싸다.)
- **Data sharing is hampered by standardization and privacy considerations**(데이터 공유가 표준화 및 개인 정보 보호 고려 사항에 의해서 방해된다.)
- **Discrimination and “right to an explanation” laws**(차별과 설명에 대한 권한)

<br/>

