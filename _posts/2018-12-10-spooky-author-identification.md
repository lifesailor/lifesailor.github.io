---
sptitle: "Spooky Author Identification"
categories:
  - kaggle
tags:
  - kaggle, machine learning, text classification

---

# Spooky Author Identification

캐글 코리아의 이유한 님이 제시한 캐글 커리큘럼과 지금 진행 중인 Quora Insincere Questions Classification 공부를 하면서, 과거의 텍스트 분류 캐글 커널을 정리하고 있습니다. 그 첫 번째로 [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification#description) Competition의 상위 1, 2등 커널을 정리했습니다. 



해당 대회는 spooky(으스스한) 문장을 보고 해당 글의 저자를 세 명(Edgar Allan Poe, Mary Shelley, HP Lovecraft) 중 한 명으로 분류하는 문제입니다. 학습 데이터는 각 문장과 저자 쌍으로 이루어진 19579개이고, 테스트 데이터는 8392개입니다. 이 대회에서 평가 지표로 multi-class logarithmic loss를 사용합니다.

```
![](assets/images/spooky-author.png "Loss Function")
```



상위 1,2등 커널은 다음의 [github](https://github.com/lifesailor/kaggle-best-kernel/tree/master/text-classification/1.spooky-author-identification)에서 확인할 수 있습니다. 

