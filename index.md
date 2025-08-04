---
layout: distill
title: "How To Scale Your Model"
subtitle: "TPU에서의 LLM 시스템 관점"
# permalink: /main/
description: "LLM 훈련은 종종 연금술처럼 느껴지지만, 모델의 성능을 이해하고 최적화하는 것은 그럴 필요가 없습니다. 이 책은 언어 모델 스케일링의 과학을 명확히 하는 것을 목표로 합니다: TPU(와 GPU)가 어떻게 작동하고 서로 어떻게 통신하는지, LLM이 실제 하드웨어에서 어떻게 실행되는지, 그리고 대규모에서 효율적으로 실행되도록 훈련과 추론 중에 모델을 어떻게 병렬화하는지. '이 LLM 훈련 비용이 얼마나 되어야 하는가' 또는 '이 모델을 직접 서빙하려면 얼마나 많은 메모리가 필요한가' 또는 'AllGather가 무엇인가'와 같은 질문을 한 적이 있다면, 이것이 유용할 것입니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

giscus_comments: true

section_number: 0

previous_section_url: ""
previous_section_name: "Part 0: 소개"

next_section_url: roofline
next_section_name: "Part 1: Rooflines"

bibliography: main.bib

citation: true

authors:
  - name: Jacob Austin
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind
  - name: Sholto Douglas
    url: "https://x.com/_sholtodouglas"
  - name: Roy Frostig
    url: "https://cs.stanford.edu/~rfrostig/"
  - name: Anselm Levskaya
    url: "https://anselmlevskaya.com/"
  - name: Charlie Chen
    url: "https://x.com/charliexychen"
  - name: Sharad Vikram
    url: "https://sharadvikram.com/"
  - name: Federico Lebron
    url: "https://fedelebron.com/"
  - name: Peter Choy
    url: "https://x.com/pchoy95"
  - name: Vinay Ramasesh
    url: "https://x.com/vinayramasesh"
  - name: Albert Webson
    url: "https://representation.ai/"
  - name: Reiner Pope<sup>*</sup>
    url: https://x.com/reinerpope

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 전체 개요
  - name: 섹션 링크

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

{% include figure.liquid path="assets/img/dragon.png" class="img-fluid" %}

딥러닝의 많은 부분이 여전히 일종의 흑마법으로 귀결되지만, 모델의 성능을 최적화하는 것은 — 심지어 대규모에서도 — 그럴 필요가 없습니다! 상대적으로 간단한 원칙들이 모든 곳에 적용됩니다 — 단일 accelerator를 다루는 것부터 수만 개까지 — 그리고 이를 이해하면 많은 유용한 일들을 할 수 있습니다:

- 모델의 각 부분이 이론적 최적에 얼마나 가까운지 대략 파악하기.
- 다양한 스케일에서 서로 다른 병렬성 방식에 대해 정보에 기반한 선택하기 (여러 디바이스에 걸쳐 계산을 분할하는 방법).
- 대형 Transformer 모델을 훈련하고 실행하는 데 필요한 비용과 시간 추정하기.
- [특정](https://arxiv.org/abs/2205.14135) [하드웨어](https://arxiv.org/abs/1911.02150) [affordances](https://arxiv.org/abs/2007.00072)를 활용하는 알고리즘 설계하기.
- 현재 알고리즘 성능을 제한하는 요소에 대한 명시적 이해를 바탕으로 하드웨어 설계하기.

**예상 배경 지식:** 우리는 여러분이 LLM과 Transformer 아키텍처에 대한 기본적인 이해를 가지고 있지만 반드시 대규모에서 어떻게 작동하는지는 알 필요가 없다고 가정합니다. LLM 훈련의 기본 사항을 알고 있어야 하며 이상적으로는 JAX에 대한 기본적인 친숙함이 있어야 합니다. 유용한 배경 읽기 자료로는 Transformer 아키텍처에 대한 [이 블로그 포스트](https://jalammar.github.io/illustrated-transformer/)와 [원본 Transformer 논문](https://arxiv.org/abs/1706.03762)이 있습니다. 또한 더 유용한 동시 및 향후 읽기 자료는 [이 목록](conclusion#further-reading)을 확인하세요.

**목표 및 피드백:** 마지막에, 여러분은 주어진 하드웨어 플랫폼에서 Transformer 모델에 대한 최적의 병렬성 방식을 추정하고, 훈련과 추론이 대략 얼마나 걸릴지 편안하게 느낄 수 있어야 합니다. 그렇지 않다면, 이메일을 보내거나 댓글을 남겨주세요! 이를 더 명확하게 만들 수 있는 방법을 알고 싶습니다.

### 왜 신경 써야 할까요?

3-4년 전이라면, 대부분의 ML 연구자들이 이 책의 내용을 이해할 필요가 없었을 것이라고 생각합니다. 하지만 오늘날 "작은" 모델조차 하드웨어 한계에 너무 가깝게 실행되어서 새로운 연구를 하려면 대규모 효율성에 대해 생각해야 합니다.<d-footnote>역사적으로, ML 연구는 시스템 혁신과 소프트웨어 개선 사이의 일종의 틱-톡 사이클을 따라왔습니다. Alex Krizhevsky는 CNN을 빠르게 만들기 위해 비정상적인 CUDA 코드를 작성해야 했지만 몇 년 안에 Theano와 TensorFlow 같은 라이브러리 덕분에 그럴 필요가 없었습니다. 여기서도 마찬가지로 이 책의 모든 내용이 몇 년 안에 추상화될 수도 있습니다. 하지만 스케일링 법칙이 우리의 모델을 하드웨어의 최전선으로 지속적으로 밀어붙이고 있으며, 가까운 미래에 최첨단 연구를 하는 것은 대형 하드웨어 토폴로지로 모델을 효율적으로 스케일링하는 방법에 대한 이해와 불가분의 관계에 있을 것 같습니다.</d-footnote> **벤치마크에서 20% 향상이 roofline 효율성에 20% 비용을 가져온다면 무의미합니다.** 유망한 모델 아키텍처들은 대규모에서 효율적으로 실행될 _수 없거나_ 아무도 그렇게 만들기 위한 작업을 하지 않기 때문에 정기적으로 실패합니다.

**"모델 스케일링"의 목표는 훈련이나 추론에 사용되는 칩 수를 늘리면서 처리량에서 비례적이고 선형적인 증가를 달성할 수 있는 것입니다.** 이를 "*strong scaling*"이라고 합니다. 추가 칩을 추가("병렬성")하면 일반적으로 계산 시간이 감소하지만, 칩 간의 통신 비용도 추가됩니다. 통신이 계산보다 오래 걸리면 "통신 제한(communication bound)"이 되어 강하게 스케일링할 수 없습니다.<d-footnote>계산 시간이 감소하면, 일반적으로 단일 칩 수준에서 병목 현상도 직면합니다. 새로운 TPU나 GPU가 초당 500조 번의 연산을 수행한다고 평가받을 수 있지만, 메모리에서 매개변수를 이동하는 데 막히면 주의하지 않으면 그것의 10분의 1만큼 쉽게 할 수 있습니다. 칩당 계산, 메모리 대역폭, 그리고 총 메모리의 상호작용은 스케일링 스토리에 중요합니다.</d-footnote> 하드웨어를 충분히 잘 이해하여 이러한 병목 현상이 어디서 발생할지 예상할 수 있다면, 이를 피하도록 모델을 설계하거나 재구성할 수 있습니다.<d-footnote>하드웨어 설계자들은 역 문제에 직면합니다: 비용을 최소화하면서 우리의 알고리즘에 충분한 계산, 대역폭, 메모리를 제공하는 하드웨어 구축하기. 이 "공동 설계" 문제가 얼마나 스트레스가 많은지 상상할 수 있습니다: 실제 칩이 사용 가능해질 때 알고리즘이 어떻게 보일지에 대해 내기를 해야 하며, 종종 2-3년 후의 일입니다. TPU의 이야기는 이 게임에서 대단한 성공입니다. 행렬 곱셈은 거의 다른 어떤 것보다 메모리 바이트당 훨씬 더 많은 FLOP을 사용한다는 점에서 독특한 알고리즘입니다(N FLOP per byte), 그리고 초기 TPU와 그들의 systolic array 아키텍처는 그들이 구축된 당시 GPU보다 훨씬 더 나은 성능/비용을 달성했습니다. TPU는 ML 워크로드를 위해 설계되었고, TensorCore를 가진 GPU도 이 틈새를 채우기 위해 빠르게 변화하고 있습니다. 하지만 신경망이 성공하지 못했거나, TPU(본질적으로 GPU보다 덜 유연함)가 처리할 수 없는 근본적인 방식으로 변화했다면 얼마나 비용이 많이 들었을지 상상할 수 있습니다.</d-footnote>

*이 책의 목표는 TPU(와 GPU) 하드웨어가 어떻게 작동하는지, 그리고 Transformer 아키텍처가 현재 하드웨어에서 잘 작동하도록 어떻게 진화했는지 설명하는 것입니다. 이것이 새로운 아키텍처를 설계하는 연구자와 현재 세대의 LLM을 빠르게 실행하기 위해 작업하는 엔지니어 모두에게 유용하기를 바랍니다.*

## 전체 개요

이 책의 전체 구조는 다음과 같습니다:

[섹션 1](roofline)은 roofline 분석과 스케일링 능력을 제한할 수 있는 요인들(통신, 계산, 메모리)을 설명합니다. [섹션 2](tpus)와 [섹션 3](sharding)은 TPU와 현대 GPU가 어떻게 작동하는지를 개별 칩으로서와 — 매우 중요하게 — 제한된 대역폭과 지연 시간을 가진 칩 간 링크를 가진 상호 연결된 시스템으로서 자세히 설명합니다. 다음과 같은 질문들에 답할 것입니다:

* 특정 크기의 행렬 곱셈이 얼마나 걸려야 하나? 언제 계산이나 메모리 또는 통신 대역폭에 의해 제한되나?
* TPU들이 어떻게 연결되어 훈련 클러스터를 형성하나? 시스템의 각 부분이 얼마나 많은 대역폭을 가지고 있나?
* 여러 TPU에 걸쳐 배열을 수집, 분산, 또는 재분배하는 데 얼마나 걸리나?
* 디바이스에 걸쳐 다르게 분산된 행렬을 어떻게 효율적으로 곱하나?

{% include figure.liquid path="assets/img/pointwise-product.gif" class="img-small" caption="<b>그림:</b> TPU가 elementwise product를 수행하는 방법을 보여주는 <a href=\"tpus\">섹션 2</a>의 다이어그램. 배열의 크기와 다양한 링크의 대역폭에 따라, 우리는 compute-bound(전체 하드웨어 계산 용량 사용) 또는 comms-bound(메모리 로딩에 의한 병목) 상태에 있을 수 있습니다." %}

5년 전 ML은 다채로운 아키텍처 환경을 가지고 있었습니다 — ConvNet, LSTM, MLP, Transformer — 하지만 이제 우리는 대부분 Transformer<d-cite key="transformers"></d-cite>만 가지고 있습니다. 우리는 Transformer 아키텍처의 모든 부분을 이해하는 것이 가치 있다고 강하게 믿습니다: 모든 행렬의 정확한 크기, 정규화가 어디서 발생하는지, 각 부분에 얼마나 많은 매개변수와 FLOP<d-footnote>FLoating point OP, 기본적으로 필요한 덧셈과 곱셈의 총 수입니다. 많은 자료에서 FLOP을 "초당 연산"을 의미하는 것으로 사용하지만, 우리는 이를 명시적으로 나타내기 위해 FLOP/s를 사용합니다.</d-footnote>이 있는지. [섹션 4](transformers)는 이 "Transformer 수학"을 신중하게 살펴보며, 훈련과 추론 모두에 대해 매개변수와 FLOP을 세는 방법을 보여줍니다. 이것은 우리 모델이 얼마나 많은 메모리를 사용할지, 계산이나 통신에 얼마나 많은 시간을 보낼지, 그리고 attention이 언제 feed-forward 블록에 비해 중요해질지 알려줍니다.

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>그림:</b> 각 행렬 곱셈(matmul)이 원 안의 점으로 표시된 표준 Transformer 레이어. 모든 매개변수(norms 제외)는 보라색으로 표시됩니다. <a href=\"transformers\">섹션 4</a>는 이 다이어그램을 더 자세히 살펴봅니다." %}

[섹션 5: 훈련](training)과 [섹션 7: 추론](inference)은 이 에세이의 핵심으로, 근본적인 질문을 논의합니다: 어떤 크기의 모델과 어떤 수의 칩이 주어졌을 때, "strong scaling" 체제에 머물기 위해 모델을 어떻게 병렬화해야 할까요? 이것은 놀랍도록 복잡한 답을 가진 간단한 질문입니다. 높은 수준에서, 여러 칩에 걸쳐 모델을 분할하는 데 사용되는 4가지 주요 병렬성 기법(**data**, **tensor**, **pipeline** 그리고 **expert**)과 메모리 요구 사항을 줄이는 다른 여러 기법들(**rematerialisation**, **optimizer/model sharding (aka ZeRO)**, **host offload**, **gradient accumulation**)이 있습니다. 우리는 여기서 이들 중 많은 것을 논의합니다.

이 섹션들이 끝날 때까지 여러분이 새로운 아키텍처나 설정에 대해 직접 선택할 수 있기를 바랍니다. [섹션 6](applied-training)과 [섹션 8](applied-inference)은 이러한 개념을 인기 있는 오픈소스 모델인 LLaMA-3에 적용하는 실용적인 튜토리얼입니다.

마지막으로, [섹션 9](profiling)와 [섹션 10](jax-stuff)은 JAX에서 이러한 아이디어 중 일부를 구현하는 방법과 문제가 발생했을 때 코드를 프로파일링하고 디버그하는 방법을 살펴봅니다.

전체에 걸쳐 우리는 여러분이 직접 작업할 문제들을 제공하려고 합니다. 모든 섹션을 읽거나 순서대로 읽어야 한다는 압박감을 느끼지 마세요. 그리고 피드백을 남겨주세요. 당분간, 이것은 초안이며 계속 수정될 것입니다. 감사합니다!

*James Bradbury와 Blake Hechtman이 이 문서의 많은 아이디어를 도출했음을 인정하고 싶습니다.*

<h3 markdown=1 class="next-section">더 이상 지체하지 말고, [여기는 TPU rooflines에 대한 섹션 1](roofline)입니다.</h3>

## 섹션 링크

*이 시리즈는 아마도 필요한 것보다 길지만, 그것이 여러분을 막지 않기를 바랍니다. 처음 세 장은 예비 지식이며 익숙하다면 건너뛸 수 있지만, 나중에 사용되는 표기법을 소개합니다. 마지막 세 부분은 실제 모델과 작업하는 방법을 설명하므로 가장 실용적으로 유용할 수 있습니다.*

**Part 1: 예비 지식**

* [**Chapter 1: Roofline 분석 간단 소개**](roofline). 알고리즘은 세 가지에 의해 제한됩니다: 계산, 통신, 메모리. 우리는 이를 사용하여 알고리즘이 얼마나 빨리 실행될지 근사할 수 있습니다.

* [**Chapter 2: TPU에 대해 생각하는 방법**](tpus). TPU는 어떻게 작동하나요? 그것이 우리가 훈련하고 서빙할 수 있는 모델에 어떤 영향을 미치나요?

* [**Chapter 3: 샤딩된 행렬과 곱셈 방법**](sharding). 여기서 우리는 우리가 좋아하는 연산인 (샤딩된) 행렬 곱셈을 통해 모델 샤딩과 multi-TPU 병렬성을 설명합니다.

**Part 2: Transformers**

* [**Chapter 4: 알아야 할 모든 Transformer 수학**](transformers). Transformer가 forward와 backward pass에서 얼마나 많은 FLOP을 사용하나요? 매개변수의 수를 계산할 수 있나요? KV cache의 크기는? 우리는 여기서 이 수학을 다룹니다.

* [**Chapter 5: 훈련을 위해 Transformer를 병렬화하는 방법**](training). FSDP. Megatron 샤딩. Pipeline 병렬성. 어떤 수의 칩이 주어졌을 때, 주어진 크기의 모델을 주어진 배치 크기로 가능한 한 효율적으로 훈련하려면 어떻게 해야 하나요?

* [**Chapter 6: TPU에서 LLaMA 3 훈련하기**](applied-training). TPU에서 LLaMA 3를 어떻게 훈련할까요? 얼마나 걸릴까요? 비용은 얼마나 될까요?

* [**Chapter 7: Transformer 추론에 대한 모든 것**](inference). 모델을 훈련한 후에는 서빙해야 합니다. 추론은 새로운 고려 사항인 지연 시간을 추가하고 메모리 환경을 변화시킵니다. 분리된 서빙이 어떻게 작동하는지와 KV cache에 대해 어떻게 생각하는지 이야기할 것입니다.

* [**Chapter 8: TPU에서 LLaMA 3 서빙하기**](applied-inference). TPU v5e에서 LLaMA 3를 서빙하는 비용은 얼마나 될까요? 지연 시간/처리량 trade-off는 무엇인가요?

**Part 3: 실용적 튜토리얼**

* [**Chapter 9: TPU 코드 프로파일링 방법**](profiling). 실제 LLM은 위의 이론만큼 간단하지 않습니다. 여기서 우리는 JAX + XLA 스택과 JAX/TensorBoard 프로파일러를 사용하여 실제 문제를 디버그하고 수정하는 방법을 설명합니다.

* [**Chapter 10: JAX로 TPU 프로그래밍하기**](jax-stuff). JAX는 계산을 병렬화하기 위한 마법 같은 API들을 제공하지만, 사용 방법을 알아야 합니다. 재미있는 예제와 실습 문제.

* [**Chapter 11: 결론과 추가 읽기**](conclusion). 마무리 생각과 TPU 및 LLM에 대한 추가 읽기.