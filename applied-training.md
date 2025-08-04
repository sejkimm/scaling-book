---
layout: distill
title: "TPU에서 LLaMA 3 훈련하기"
# permalink: /main/
description: "이전 섹션에서 배운 내용을 사용하여 TPU v5p에서 LLaMA 3 모델을 어떻게 훈련할지 자세히 살펴보겠습니다. 모델들의 크기는 얼마나 될까요? 다양한 구성에서 훈련 비용은 어떻게 될까요? 어떻게 샤딩될까요? 이전 섹션들이 실제 모델에 어떻게 적용되는지 몇 가지 대략적인 추정을 통해 살펴보겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 6

previous_section_url: "../training-kr"
previous_section_name: "Part 5: Training"

next_section_url: ../inference-kr
next_section_name: "Part 7: Inference"

bibliography: main.bib

giscus_comments: true

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
  - name: "LLaMA 3는 어떻게 생겼나요?"
  - name: "매개변수와 FLOP 계산하기"
  - name: "훈련을 위한 LLaMA 3-70B 샤딩 방법"
  - name: "실습 문제"

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

_이번 섹션의 목표는 이전 섹션의 결과를 매우 실용적인 문제에 적용하는 것입니다: LLaMA 3 모델 패밀리 훈련하기. 이전 섹션들과 달리 여러분이 직접 많은 작업을 해보시길 바랍니다. 이런 이유로, 각 섹션의 답을 숨겨두었으니 먼저 스스로 답해보실 수 있습니다. 펜을 들고 손으로 직접 해보세요!_

### LLaMA 3는 어떻게 생겼나요?

LLaMA-3 모델 패밀리<d-cite key="llama3"></d-cite>는 3개의 주요 모델을 포함합니다: LLaMA 3 8B, 70B, 405B. 우리는 주로 70B에 집중하고, 8B와 405B는 마지막 문제 섹션에서 탐색하도록 남겨두겠습니다. LLaMA [HuggingFace 페이지](https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json)에서 가져온 LLaMA 3-70B의 아키텍처는 다음과 같습니다.

| **hyperparam**              | **value** |
| --------------------------- | --------- |
| $$n_\text{layers}$$ (L)     | 80        |
| $$d_\text{model}$$ (D)      | 8,192     |
| $$d_{ff}$$ (F)              | 28,672    |
| $$n_\text{heads}$$ (N)      | 64        |
| $$n_\text{kv_heads}$$ (K)   | 8         |
| $$d_\text{qkv}$$ (H)        | 128       |
| $$n_\text{embeddings}$$ (V) | 128,256   |

이를 찾는 것이 얼마나 쉬운지 강조하기 위해, 다음은 매핑과 함께 config 자체입니다:

{% include figure.liquid path="assets/img/llama-json.png" class="img-fluid" %}

_많은 다른 오픈소스 LLM에 대해 이런 숫자들로 큰 표를 만들어두면, 그들이 내린 설계 결정을 빠르게 비교할 수 있어 유용합니다._

### 매개변수와 FLOP 계산하기

**질문:** 이 표에서 LLaMA 3-70B 매개변수 수를 계산할 수 있을까요? 🤫 [Section 4](../transformers-kr)의 내용을 적용해서 70B를 얻을 수 있는지 봅시다!

| param            | formula                                                                                                                                           | count                                                        |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| FFW params       | d_model * d_ff * 3 (for gelu + out-projection) * n_layers                                                                                         | 8,192 * 8,192 * 3.5 * 3 * 80 = **56.3e9**                    |
| Vocab params     | 2 (input and output embeddings) * n_embeddings * d_model                                                                                          | 2 * 128,256 * 8,192 = **2.1e9**                              |
| Attention params | n_layers * [ 2 (for q embedding and concatenated output projection) * d_model * n_heads * d_qkv + 2 (for k and v) * d_model * n_kv_heads * d_qkv] | 80 * (2 * 8,192 * 64 * 128 + 2 * 8,192 * 8 * 128) = **12e9** |
|                  |                                                                                                                                                   | 56.3e9 + 2.1e9 + 12e9 = **70.4e9**                           |

훌륭합니다! 예상한 숫자를 얻었습니다. 예상대로 FFW 매개변수가 전체 매개변수 수를 완전히 지배하지만, attention도 무시할 수 없습니다.

<p markdown=1 class="takeaway">**핵심 요점**: MLP 블록의 3개 큰 weight matrix가 Transformer의 다른 모든 배열보다 훨씬 커서, 모델 메모리나 FLOP에 대해 추론할 때 일반적으로 다른 모든 매개변수를 거의 무시할 수 있습니다. LLaMA 3-70B의 경우, 70B 매개변수 중 56B를 차지합니다.</p>

이제 FLOP를 살펴보겠습니다! *[Section 4](../transformers-kr)에서 훈련에 대한 일반적인 규칙을 기억하세요.*

**질문:** LLaMA-3는 훈련 스텝당 토큰당 몇 개의 FLOP를 수행할까요? _이는 전체 훈련 과정이 얼마나 비쌀지 결정하는 데 도움이 됩니다._

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: [Section 4](../transformers-kr)에서 보여준 바와 같이, 토큰당 대략 $$6 \cdot \text{param count}$$ FLOP를 수행하므로, 여기서는 대략 `6 * 70e9 = 4.2e11` FLOP / token입니다. 스텝당 토큰당 약 0.5 TFLOP입니다. compute-bound라고 가정하면, 완벽한 FLOP 활용을 가정할 때 단일 TPU v5p 칩에서 대략 `4.2e11 / 4.59E+14 = 1ms`가 걸려야 합니다.

{% enddetails %}

**질문:** LLaMA 3는 약 15조 토큰에 대해 훈련되었습니다. 총 몇 개의 FLOP인가요?

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 간단합니다. `4.2e11 * 15e12 = 6.3e24 FLOPs` 총합입니다. 6.3 yottaFLOP입니다. 엄청 많습니다! 단일 TPU에서는 `6.3e24 / 4.59E+14 = 435년`이 걸릴 것입니다. 이것도 엄청 많습니다!

{% enddetails %}

**질문:** 16x20x28 = 8960개 칩을 가진 전체 TPU v5p pod에서 훈련하고 싶다고 해봅시다. compute-bound라고 가정할 때, bfloat16에서 40% MFU로 훈련하는 데 얼마나 걸릴까요?

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 각 TPU v5p가 초당 4.59e14 FLOP를 수행할 수 있다는 것을 알고 있습니다. 40% MFU에서, 약 `T = 6.3e24 / (8960 * 4.59e14 * 0.4) = 3.8e6초`가 걸릴 것입니다. **약 44일입니다!** 실제로 40% MFU를 달성할 수 있다고 가정하면 상당히 합리적입니다.

{% enddetails %}

**질문:** LLaMA 3-70B는 약 4M 토큰의 배치 크기로 사전 훈련되었습니다. 이 배치 크기로 훈련하려면 최소 몇 개의 TPU가 필요한가요? _bfloat16 매개변수와 float32 optimizer state를 가정하고, layer당 4번 gradient를 체크포인트한다고 가정할 수 있습니다._

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 이 질문은 주로 메모리 사용량에 관한 것입니다. 사용 가능한 계산에 대한 유일한 엄격한 제약이기 때문입니다. 훈련 중에는 HBM의 세 가지 주요 용도가 있습니다: 모델 매개변수, optimizer state, gradient checkpoint입니다. bfloat16 weight, float32 optimizer state, _매우_ 보수적인 gradient checkpointing 방식(layer당 4번)을 가정하면:

| **Params** | 2 * 70GB | ~140GB |
| **Optimizer State** | 8 * 70GB | ~560GB |
| **Gradient Checkpoints** | 2 * 8192 * 4e6 * 4 * 80 | ~20.9TB |
| **Total**                |                         | ~21.6TB |

여기서 총합은 약 21.6TB입니다. 매우 보수적인 checkpointing 방식에서도 gradient checkpointing이 메모리 상황을 강하게 지배한다는 것을 알 수 있습니다. 기술적으로는 layer당 1개 checkpoint로 가거나 microbatching을 할 수 있지만, 이는 합리적인 상황입니다. 이러한 가정 하에서, 각 TPU v5p가 96GB의 HBM을 가지므로, `21.6e12 / 96e9 = 225`개의 TPU가 필요합니다. 사실 그리 많지 않습니다!

*왜 이렇게 하지 않을까요?* 훈련하는 데 `44일 * 8960 / 225 = 1752일`이 걸리기 때문입니다. 거의 4년입니다. **엄청 많습니다.** 여전히, 이는 메모리에 바운드되어서가 아니라 추가 FLOP가 필요해서 이런 큰 클러스터를 사용한다는 것을 분명히 합니다.

{% enddetails %}

**질문:** 위 질문과 같은 가정 하에서, 8960개의 TPU v5p 칩을 사용한다면 칩당 얼마나 많은 메모리를 사용할까요?

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 총 메모리는 여전히 약 21.6TB이므로, 칩당 약 2.4GB를 사용할 것이며, 이는 기본적으로 아무것도 아닙니다. 훨씬 더 적극적인 checkpointing을 한다면, 예를 들어 layer당 12개 checkpoint라도, 칩당 8GB에 불과할 것입니다. 이런 규모에서 훈련 중에는 메모리 바운드와는 거리가 멉니다.

{% enddetails %}

<p markdown=1 class="takeaway">**핵심 요점**: 매우 큰 모델이라도 매우 작은 토폴로지에서 기술적으로 훈련하는 것이 가능하지만, 오랜 시간이 걸릴 것입니다. 훈련 실행의 총 FLOP를 계산할 수 있으면 적당한 MFU와 알려진 토폴로지를 가정하여 훈련 시간을 대략적으로 계산할 수 있습니다.</p>

### 훈련을 위한 LLaMA 3-70B 샤딩 방법

위의 설정을 유지하고 8960개 칩의 TPU v5p pod에서 4M 토큰 배치 크기(배치당 길이 4096의 1024개 시퀀스)로 LLaMA 3-70B를 훈련하고 싶다고 해봅시다. 이 모델에 대한 최적의 샤딩 전략이 무엇인지 논의해 보겠습니다.

**질문:** 위 가정 하에서, FSDP만으로 모델을 훈련할 수 있을까요? 시작하기 위해, sequence/context parallelism은 할 수 없다고 해봅시다. _이것이 여러분의 첫 번째 아이디어여야 합니다. 간단하고 작동한다면 추가 통신이 없을 것이기 때문입니다._

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 이 답은 약간 교조적일 것입니다. 위에서 언급했듯이, LLaMA 3-70B는 처음에 길이 4K의 시퀀스로 훈련되므로, 4M 토큰의 배치 크기는 *시퀀스 배치 크기* 1024를 제공합니다. 즉, data parallelism을 수행할 수 있는 시퀀스가 1024개뿐이므로, 실제로는 최대 1024개 칩까지만 순수한 data parallelism/FSDP를 할 수 있습니다. 따라서 "추가 통신 없는 완전 data parallelism"이라는 단순한 의미에서 답은 아니오입니다. 다음 질문이 이것의 약간 덜 교조적인 버전에 답할 것입니다.

{% enddetails %}

**질문:** sequence 샤딩을 하지 않는다는 요구사항을 완화해 봅시다. 배치 _와_ 시퀀스 axis 모두에 대해 FSDP를 할 수 있다면, 8960개 칩에서 FSDP만으로 LLaMA 3-70B를 훈련할 수 있을까요?

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 이제 sequence/context parallelism도 할 수 있게 되었으므로, 훨씬 더 스케일업할 수 있습니다. 먼저 디바이스당 배치 크기를 계산해 봅시다. 8960-way FSDP를 한다면, TPU당 배치 크기는 `4 * 1024 * 1024 / 8960 = 468 토큰`입니다. 이전 섹션에서 $$\text{per device batch size} < 2550 / n_\text{axes}$$일 때 FSDP로 ICI-bound가 된다는 것을 알고 있습니다. 전체 3D pod로 여기서 3개 axis를 할당할 수 있으므로, 하한은 850이 되고, 우리는 그것보다 훨씬 아래에 있습니다. **따라서 3개 axis로도 답은 아니오입니다. 확실히 통신 바운드가 될 것입니다.**

{% enddetails %}

**질문:** 이제 혼합 tensor parallelism과 FSDP를 살펴봅시다. compute-bound를 유지할 수 있는 조합이 존재할까요? 그렇다면 얼마나 많은 FSDP와 tensor parallelism을 해야 할까요?

{% details 생각해보신 후 답을 보려면 여기를 클릭하세요! %}

**답**: 먼저 이것이 맞는지 확인해 봅시다. 칩당 배치 크기가 $$2 \cdot 2550^2 / F = 453$$보다 작으면 comm-bound가 된다는 것을 알고 있습니다. 위에서 본 바와 같이, 우리는 이것보다 약간 위에 있습니다. 훌륭합니다! 이제 최적의 FSDP 양을 선택하기 위해 공식을 사용할 수 있습니다:

$$X_{opt} = \sqrt{\frac{2BN}{F}} = \sqrt{\frac{2 \cdot 4.19e6 \cdot 8960}{28672}} = 1618$$

2의 합리적인 배수로 반올림하면, 대략 2048-way FSDP와 4-way model parallelism이 나옵니다. 잘 작동할 것입니다!

{% enddetails %}

<p markdown=1 class="takeaway">**핵심 요점**: 통신 바운드 없이 전체 TPU v5p pod에서 4M 토큰 배치 크기로 LLaMA-3를 data parallelism (1024-way), sequence parallelism (2-way), tensor parallelism (4-way)의 혼합으로 훈련할 수 있습니다. 순수한 FSDP나 FSDP + sequence parallelism을 시도하면 comm-bound가 될 것입니다. 이전 섹션에서 만든 방정식은 매우 실용적입니다.</p>

## 실습 문제

**문제 1 [LLaMA 70B를 더 많은 칩으로 스케일링]:** 같은 배치 크기로 4개 pod에서 LLaMA 3-70B를 훈련하고 싶다고 해봅시다. 어떤 병렬화 방식을 사용할까요? compute 또는 통신 바운드가 될까요? 훈련하는 데 대략 얼마나 걸릴까요? *올바른 roofline bound를 사용하세요.*

**문제 2 [LLaMA 405B]:**

(a) LLaMA 3-405B [config](https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json)을 사용하여, 위와 같이 모든 핵심 hyperparameter가 포함된 표를 작성하세요. 이 모델의 총 매개변수 수는 얼마인가요? 훈련 스텝당 몇 개의 FLOP인가요? 15T 토큰에 대해 훈련한다면 몇 개의 FLOP를 수행할까요?

(b) 8개의 TPU v5p pod에서 훈련하고 싶다고 가정하세요. 어떤 병렬화 방식을 사용할까요? 훈련하는 데 얼마나 걸릴까요? compute 또는 comm bound가 될까요?

<h3 markdown=1 class="next-section">Section 6은 여기까지입니다. Transformer 추론에 관한 Section 7은 [여기를 클릭하세요](../inference-kr).</h3>