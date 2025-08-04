# 모델 스케일링 방법

> Claude Sonnet 4 를 통해 [원본 문서](https://github.com/jax-ml/scaling-book)를 한국어로 번역하였고, 현재 인간 리뷰어가 검토중인 문서입니다.

이 책은 TPU에서 LLM을 확장하는 기술을 알기 쉽게 설명하는 것을 목표로 합니다. TPU가 어떻게 작동하는지, LLM이 실제로 어떻게 대규모로 실행되는지, 그리고 훈련과 추론 중에 통신 병목 현상을 피하는 병렬 처리 방식을 선택하는 방법을 설명하려고 합니다. 이 책은 <https://jax-ml.github.io/scaling-book> 에서 확인할 수 있습니다.

### 감사의 말

이 책은 Google DeepMind의 Jacob Austin, Sholto Douglas, Roy Frostig, Anselm Levskaya, Charlie Chen, Sharad Vikram, Federico Lebron, Peter Choy, Vinay Ramasesh, Albert Webson이 작성했습니다. 많은 아이디어는 James Bradbury와 Reiner Pope가 처음 도출했습니다.

웹사이트는 <https://github.com/alshedivat/al-folio> 및 Distill 팀이 만든 Distill 스타일 Jekyll 테마를 사용합니다. 감사합니다!

### 로컬 실행

이 repository를 로컬에서 빌드하려면 다음을 실행하세요:

```
git clone https://github.com/jax-ml/scaling-book.git
cd scaling-book
bundle install
bundle exec jekyll serve
```

Mac OS에서 실행하려면 발생하는 오류에 따라 다음 중 일부를 실행해야 할 수도 있습니다: `brew install imagemagick`, `pip install jupyter`, `brew install ruby`, 또는 `git config safe.bareRepository all`. jekyll serve를 성공적으로 실행하면 책은 `localhost:4000/scaling-book`에서 확인할 수 있습니다.

GitHub Pages 사이트에 배포하려면 (repository 쓰기 권한 필요) `sh bin/deploy`를 실행하세요. 실행하는 데 약 3분이 걸립니다.

### 기여 및 연락처

문제를 발견하거나 질문이 있으면 웹사이트 자체에 댓글을 남기거나 (Giscus 제공) GitHub discussion에 남겨주세요. 기여하고 싶다면 PR을 자유롭게 보내주세요. jaaustin [at] google [dot] com으로 이메일을 보낼 수도 있습니다.

GitHub에서 기여하려면 Google "Contributor License Agreement" (CLA)에 서명해야 합니다. 여기서 할 수 있습니다: <https://cla.developers.google.com/clas>.

### 인용

학술적 맥락에서 인용할 때는 다음과 같이 이 작업을 인용해 주세요:

```Austin et al., "How to Scale Your Model", Google DeepMind, online, 2025.```

BibTeX 인용

```
@article{scaling-book,
  title = {How to Scale Your Model},
  author = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and Lebron, Federico and Choy, Peter and Ramasesh, Vinay and Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  howpublished = {Online},
  note = {Retrieved from https://jax-ml.github.io/scaling-book/},
  year = {2025}
}
```

![dragon](assets/img/dragon.png)

*이 책은 원래 Dreamworks 영화의 이름을 따서 "How To Scale Your Dragon"이라고 불렸으며, 그래서 용 이미지가 사용되었습니다.*
