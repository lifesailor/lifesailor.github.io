---
title: "Spooky Author Identification"
categories:
  - kaggle
tags:
  - kaggle
  - machine-learning
  - natural-language-processing
---

캐글 코리아의 이유한 님이 제시한 캐글 커리큘럼과 지금 진행 중인 Quora Insincere Questions Classification 공부하면서 과거의 텍스트 분류 캐글 커널을 정리하고 있습니다. 그 첫 번째로 [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification#description) Competition의 상위 1, 2등 커널을 정리했습니다.



Spooky Author Identification 대회의 과제는 spooky(으스스한) 문장을 보고 해당 글의 저자를 세 명(Edgar Allan Poe, Mary Shelley, HP Lovecraft) 중 한 명으로 정하는 것입니다. 학습 데이터는 각 문장과 저자 쌍으로 이루어진 19579개이고, 테스트 데이터는 8392개입니다. 평가 지표로는 multi-class logarithmic loss를 사용했습니다. 



![](/assets/images/spooky-author-identification/evaluation.png)



train 데이터의 처음 5줄은 다음과 같습니다. 



![](/assets/images/spooky-author-identification/example.png)



EAP - Edgar Allen Poe
HPL - HP Lovecraft
MWS - Mary Shelley



이 대회는 Playground 대회로 Competition 점수를 주지 않고 많은 사람들에게 도움이 되는 Kernel과 Discussion을 제공한 사람들에게 상금을 수여헀습니다. 해당 대회에 대한 자세한 내용 및 상위 1,2등 커널은 다음 [github](https://github.com/lifesailor/kaggle-best-kernel/tree/master/text-classification/1.spooky-author-identification) 또는 [kaggle](https://www.kaggle.com/c/spooky-author-identification#description)에서 확인할 수 있습니다.

