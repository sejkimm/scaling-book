---
layout: distill
title: "Transformer 훈련을 위한 병렬화 방법"
# permalink: /main/
description: "LLM 훈련에서 사용되는 네 가지 주요 병렬화 방식을 다룹니다: data parallelism, fully-sharded data parallelism (FSDP), tensor parallelism, pipeline parallelism. 각각에 대해 언제 통신이 병목이 되는지 계산합니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 5

previous_section_url: "../transformers-kr"
previous_section_name: "Part 4: Transformers"

next_section_url: ../applied-training-kr
next_section_name: "Part 6: Training LLaMA"

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
  - name: "스케일링이란 무엇인가?"
  - subsections:
    - name: "Data Parallelism"
    - name: "Fully-Sharded Data Parallelism (FSDP)"
    - name: "Tensor Parallelism"
    - name: "혼합 FSDP와 Tensor Parallelism"
    - name: "Pipelining"
    - name: "Pod 간 스케일링"
  - name: "TPU에서 LLM 훈련의 핵심 요점"
  - name: "연습 문제"
  - name: "부록"

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

## 스케일링이란 무엇인가?

"모델 스케일링"의 목표는 훈련이나 추론에 사용되는 칩의 수를 증가시켜 비례적인 선형 처리량 증가를 달성하는 것입니다(*strong scaling*이라고 합니다). 단일 칩 성능이 메모리 대역폭과 FLOP 간의 트레이드오프에 의존하는 반면, 클러스터 수준 성능은 유용한 FLOP과 겹치면서 칩 간 통신을 숨기는 것에 의존합니다. 이는 칩 수를 늘리면 통신 부하가 증가하는 동시에 이를 숨길 수 있는 디바이스당 계산량이 줄어들기 때문에 간단하지 않습니다. [Section 3](../sharding-kr)에서 본 것처럼, 샤딩된 matrix multiplication은 종종 TPU의 유용한 작업을 차단할 수 있는 비싼 AllGather나 ReduceScatter를 필요로 합니다. 이 섹션의 목표는 이것들이 언제 *너무 비싸지는지* 알아내는 것입니다.

이 섹션에서는 네 가지 일반적인 병렬화 방식을 논의합니다: (순수한) **data parallelism, fully-sharded data parallelism** (FSDP / ZeRO sharding), **tensor parallelism** (model parallelism라고도 알려진), 그리고 (간략히) **pipeline parallelism**. 각각에 대해 발생하는 통신 비용과 그 비용이 언제 계산 비용을 병목시키기 시작하는지 보여드리겠습니다.<d-footnote>통신 한계에 집중하겠습니다. 메모리 용량 제약이 중요하지만, 일반적으로 rematerialization (activation checkpointing)과 사전 훈련 중 매우 많은 수의 칩을 사용할 때는 제약이 되지 않습니다. 또한 MoE의 expert parallelism은 여기서 다루지 않습니다. 이는 설계 공간을 상당히 확장하며, 오직 dense Transformer의 기본 케이스만 다룹니다.</d-footnote> 이 섹션에서는 충분히 큰 단일 칩 배치 크기가 있는 한 HBM에서 MXU로의 데이터 전송이 이미 계산과 겹쳐지므로, 칩 간 통신 비용에만 집중할 수 있습니다.

이 섹션 전체에서 계산을 단순화하기 위해 다음 표기법을 사용하겠습니다.

| 표기법 | 의미 (모델 매개변수)                                             |
| :------- | :--------------------------------------------------------------------- |
| D        | **d**<sub>model</sub> ( hidden dimension/residual stream dim)      |
| F        | **d**<sub>ff</sub> (feed-forward dimension)                        |
| B        | Batch dimension (배치의 토큰 수; 총합, 디바이스당이 아님) |
| T        | Sequence length                                                        |
| L        | 모델의 layer 수                                          |

| 표기법 | 의미 (하드웨어 특성)                                                                 |
| :------- | :------------------------------------------------------------------------------------------------ |
| C        | 칩당 FLOPS/s                                                                                  |
| W        | Network bandwidth (양방향, 종종 $W_{\text{ici}}$ 또는 $W_{\text{dcn}}$로 첨자 표기) |
| X        | mesh axis X를 따른 칩 수                                                                 |
| Y        | Y로 표시된 대체 mesh axis를 따른 칩 수                                           |
| Z        | Z로 표시된 세 번째 mesh axis를 따른 칩 수                                                |

단순화를 위해 **Transformer를 MLP 블록의 스택으로 근사**하겠습니다 — attention은 [Section 4](../transformers-kr)에서 본 것처럼 더 큰 모델의 경우 FLOP의 비교적 작은 부분입니다. 또한 gating matmul을 무시하여 각 layer에 대해 다음과 같은 간단한 구조를 남깁니다:

{% include figure.liquid path="assets/img/simple-transformer.png" class="img-fluid" caption="<b>Figure:</b> 단순화된 Transformer layer. 각 FFW 블록을 두 개의 matrix stack으로 취급합니다: <b>W<sub>in</sub></b>: <code>bf16[D, F]</code> (up-projection)와 <b>W<sub>out</sub></b>: <code>bf16[F, D]</code> (down-projection), 입력 <b>In</b>: <code>bf16[B, D]</code>." %}

다음은 우리가 논의할 4가지 병렬화 방식입니다. 각 방식은 위 다이어그램에서 **In**, **W<sub>in</sub>, W<sub>out</sub>, Out**에 대한 샤딩으로 고유하게 정의될 수 있습니다.

**1. Data parallelism:** *activation은 배치를 따라 샤딩되고, 매개변수와 optimizer state는 각 디바이스에 복제됩니다. 통신은 역방향 패스 중에만 발생합니다.*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**2. Fully-sharded data parallelism (FSDP 또는 ZeRO-3):** *activation은 배치를 따라 샤딩되고(순수한 data parallelism처럼), 매개변수는 동일한 mesh axis를 따라 샤딩되며 순방향 패스에서 사용 직전에 AllGather됩니다. Optimizer state도 배치를 따라 샤딩됩니다. 중복된 메모리를 줄입니다.*

$$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

**3. Tensor parallelism (Megatron sharding 또는 model parallelism라고도 함):** *activation은 D ($d_\text{model}$)를 따라 샤딩되고, 매개변수는 F ($d_{ff}$)를 따라 샤딩됩니다. 각 블록 전후에 activation을 AllGather하고 ReduceScatter합니다. FSDP와 호환됩니다.*

$$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$

**4. Pipeline parallelism:** *weight은 layer dimension을 따라 샤딩되고, activation은 microbatch되어 layer dimension을 따라 rolled됩니다. 파이프라인 스테이지 간 통신은 최소한입니다(단일 홉을 통해 activation만 이동). 표기법을 남용하면:*

$$\text{In}[L_Z, B, D][i] \cdot_D W_\text{in}[L_Z, D, F][i] \cdot_F W_\text{out}[L_Z, F, D][i] \rightarrow \text{Out}[L_Z, B, D_Y][i]$$

### Data Parallelism

**문법:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D]$$

**모델이 아주 작은 배치 크기(>240 토큰, compute-bound가 되도록)로도 단일 칩에 맞는 경우, 항상 단순한 data parallelism을 사용해야 합니다.** 순수한 data parallelism은 배치 크기보다 작은 수의 TPU에 activation을 분할합니다. 순방향 패스는 통신을 포함하지 않지만, 매 스텝 끝에서 각각은 매개변수를 업데이트하기 전에 동기화하기 위해 **gradient에 대한 AllReduce를 수행합니다.**

{% include figure.liquid path="assets/img/data-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 순수한 data parallelism의 다이어그램 (순방향 패스). activation(왼쪽)은 배치 dimension을 따라 완전히 샤딩되고 weight은 완전히 복제되므로, 각 TPU는 weight의 동일한 복사본을 가집니다. 이는 weight의 총 메모리가 N만큼 증가함을 의미하지만, 순방향 패스에서는 통신이 필요하지 않습니다." %}

{% details 순방향 및 역방향 패스의 전체 알고리즘입니다. 간결함을 위해 dL/dOut을 dOut으로 표기합니다. %}

<div markdown=1 class="algorithm">

**순수한 Data Parallelism 알고리즘:**

**순방향 패스:** Loss[B<sub>X</sub>]를 계산해야 함

1.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F]
2.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
3.  Loss[B<sub>X</sub>] = ...

**역방향 패스:** dW<sub>out</sub>[F, D], dW<sub>in</sub>[D, F]를 계산해야 함

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D] = **AllReduce**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*critical path에 있지 않음, 비동기적으로 수행 가능*)
4.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D]
5.  dW<sub>in</sub>[D, F] {U<sub>X</sub>} = In[B<sub>X</sub>, D] \*<sub>B</sub> dTmp[B<sub>X</sub>, F]
6.  dW<sub>in</sub>[D, F] = **AllReduce**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) (*critical path에 있지 않음, 비동기적으로 수행 가능*)
7.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*이전 layer에 필요*)

</div>

loss function의 세부사항을 무시하고 $\text{Tmp} = W_\text{in} \cdot \text{In}$로 축약합니다. 최종 loss가 평균 **AllReduce**(Loss[B<sub>X</sub>])이지만, weight gradient를 평균낸 때만 역방향 패스에서 AllReduce를 계산하면 됩니다.

{% enddetails %}

순방향 패스에는 통신이 없으며 — **모든 것이 역방향 패스에 있습니다**! 역방향 패스는 또한 AllReduce가 "critical path"에 있지 않다는 훌륭한 특성을 가지고 있어, 각 AllReduce는 편리한 때에 수행될 수 있으며 후속 연산을 차단하지 않습니다. 총 통신 비용은 총 계산 비용을 초과하면 _여전히 병목이 될 수 있지만_, 구현 관점에서는 훨씬 관대합니다. model/tensor parallelism은 이런 특성을 갖지 않는다는 것을 보게 될 것입니다.

**왜 이렇게 하는가?** 순수한 data parallelism은 배치 dimension에 걸쳐 activation을 분할하여 activation 메모리 압박을 줄이고, 배치 dimension을 분할할 더 많은 칩이 있는 한 배치 크기를 거의 임의로 증가시킬 수 있게 합니다. 특히 훈련 중 activation이 종종 메모리 사용량을 지배할 때 이는 매우 도움이 됩니다.

**왜 이렇게 하지 않는가?** 순수한 data parallelism은 모델 매개변수나 optimizer state의 메모리 압박을 줄이는 데 도움이 되지 않으므로, 매개변수 + optimizer state가 단일 TPU에 맞지 않는 대규모 흥미로운 모델에서는 순수한 data parallelism이 거의 유용하지 않습니다. 규모를 알려주기 위해, bf16 매개변수와 Adam을 사용한 fp32 optimizer state로 훈련한다면<d-footnote>Adam은 매개변수, 1차 및 2차 누적기를 저장합니다. 매개변수는 bfloat16이고 optimizer state는 float32이므로, 매개변수당 `2 + 8 = 10` 바이트가 됩니다.</d-footnote>, 맞출 수 있는 가장 큰 모델은 $$\text{TPU memory} / 10$$ 매개변수를 가지므로, 예를 들어 96GB HBM을 가진 TPUv5p pod에서 순수한 data parallelism으로는 약 9B 매개변수입니다.

<p markdown=1 class="takeaway">**핵심 요점**: Adam과 순수한 data parallelism으로 훈련할 수 있는 가장 큰 모델은 $$\text{num_params} = \text{HBM per device} / 10$$입니다. TPU v5p의 경우 대략 9B 매개변수입니다.<d-footnote>이는 gradient checkpoint를 포함하지 않으므로 실제로는 유용하지 않을 것입니다. 이는 1개 토큰 배치를 가진 절대적 하한입니다.</d-footnote></p>

*실제 모델에서 훈련 중에 이를 유용하게 만들려면, 적어도 부분적으로 모델 매개변수나 optimizer를 샤딩해야 합니다.*

**언제 통신이 병목이 되는가?** 위에서 볼 수 있듯이, layer당 두 개의 AllReduce가 있으며, 각각 크기는 $$2DF$$ (bf16 weight의 경우)입니다. data parallelism이 언제 통신 바운드가 되는가?

위 표에서와 같이, $C$ = 칩당 FLOP, $W_{\text{ici}}$ = **양방향** 네트워크 대역폭, $X$ = 배치가 분할되는 샤드 수라고 하겠습니다<d-footnote>이 분할이 ICI mesh를 통해 수행된다고 가정하므로, 관련 네트워크 대역폭은 $W_\text{ici}$입니다.</d-footnote>. 관련 matmul을 수행하는 데 필요한 시간 $$T_\text{math}$$와 필요한 통신 시간 $$T_\text{comms}$$를 계산해 보겠습니다. 이 병렬화 방식은 순방향 패스에서 통신이 필요하지 않으므로, 역방향 패스에 대해서만 이 수량들을 계산하면 됩니다.

*통신 시간:* 이전 섹션에서 1D mesh에서 AllReduce를 수행하는 데 필요한 시간은 AllReduce되는 배열의 총 바이트와 ICI 대역폭 $W_\text{ici}$에만 의존한다는 것을 알고 있습니다. 구체적으로 AllReduce 시간은 $2 \cdot \text{total bytes} / W_\text{ici}$입니다. $W_\text{in}$과 $W_\text{out}$ 모두에 대해 AllReduce가 필요하므로, layer당 2개의 AllReduce가 있습니다. 각 AllReduce는 weight matrix, 즉 $DF$ 매개변수의 배열, 또는 $2DF$ 바이트입니다. 이를 모두 합치면, 단일 layer에서 AllReduce의 총 시간은

$$\begin{align}
T_\text{comms} &= \frac{2 \cdot 2 \cdot 2 \cdot D \cdot F}{W_\text{ici}}. \\
\end{align}$$

*Matmul 시간:* 각 layer는 순방향 패스에서 2개의 matmul, 또는 역방향 패스에서 4개의 matmul로 구성되며, 각각 $2(B/X)DF$ FLOP가 필요합니다. 따라서 역방향 패스에서 단일 layer에 대해 다음과 같습니다:

$$\begin{align}
T_\text{math} &= \frac{2 \cdot 2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
\end{align}$$

겹치므로, layer당 총 시간은 이 두 수량의 최댓값입니다:

$$\begin{aligned}
T &\approx \max(\frac{8 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{8 \cdot D \cdot F}{W_\text{ici}}) \\
T &\approx 8 \cdot D \cdot F \cdot \max(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}})
\end{aligned}$$

$$T_\text{math}/T_\text{comms} > 1$$일 때, 즉 다음 조건일 때 compute-bound가 됩니다:

$$\begin{align}
\frac{B}{X} > \frac{C}{W_\text{ici}}.
\end{align}$$

요점은 data parallelism으로 compute-bound 상태를 유지하려면, 디바이스당 배치 크기 $$B / X$$가 ICI operational intensity인 $C / W_\text{ici}$를 초과해야 한다는 것입니다. 이는 궁극적으로 계산 시간이 디바이스당 배치 크기에 따라 스케일되는 반면, 통신 시간은 (모델 weight을 전송하므로) 이 수량과 독립적이라는 사실의 결과입니다. $B > C/W_\text{ici}$ 조건이 단일 디바이스 compute-bound 규칙 $B > 240$과 유사함을 주목하세요. 그 경우에도 규칙은 계산 시간이 배치 크기에 따라 스케일되는 반면 데이터 전송 크기가 ($B \ll F, D$ 영역에서) 배치 크기와 독립적이라는 사실에서 나왔습니다.

실제 숫자를 넣어 규모를 알아보겠습니다. TPUv5p의 경우, `C=4.6e14`이고 ICI를 통한 1D data parallelism에 대해 `W=2 * 9e10`이므로, **통신 바운드를 피하려면 칩당 배치 크기가 최소 2,550이어야 합니다**. 여러 축에 걸쳐 data parallelism을 할 수 있으므로, TPUv5p pod의 세 축을 모두 순수한 data parallelism에 전용한다면, 대역폭 $W_\text{ici}$를 3배 늘리고 TPU당 BS=850 또는 pod당 배치당 7.6M 토큰(8960 칩 중)으로 줄일 수 있습니다! **이는 순수한 data parallelism으로 병목이 되기가 상당히 어렵다는 것을 알려줍니다!**

<p markdown=1 class="takeaway">**Context parallelism에 대한 참고:** 이 섹션 전체에서 $B$를 토큰 단위의 총 배치 크기를 의미하는 데 사용합니다. 하지만 분명히 우리의 배치는 각각 $T$ 토큰의 $K$ 시퀀스로 구성되는데, 어떻게 이렇게 할 수 있을까요? MLP에 관한 한, *토큰은 토큰입니다*! 같은 배치에 속하든 두 개의 다른 배치에 속하든 상관없습니다. 따라서 우리는 배치와 시퀀스 dimension 모두에 대해 data parallelism을 할 수 있습니다: 이를 context parallelism 또는 sequence parallelism이라고 부르지만, 단순히 또 다른 종류의 data parallelism으로 생각할 수 있습니다. Attention은 일부 cross-sequence 계산을 하므로 MLP보다 까다롭지만, attention 중에 KV나 Q를 gather하고 FLOP와 comm을 신중하게 겹치는 것으로 처리할 수 있습니다(일반적으로 "ring attention"이라고 하는 것을 사용). 이 섹션 전체에서 시퀀스 dimension을 완전히 무시하고 어느 정도의 배치 또는 시퀀스 병렬화를 가정하겠습니다.</p>


### Fully-Sharded Data Parallelism (FSDP)

**문법:** $$\text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D]$$

Fully-sharded data parallelism (종종 FSDP 또는 ZeRO-sharding<d-cite key="zero"></d-cite>라고 함)은 모델 optimizer state와 weight을 data parallel 샤드에 걸쳐 분할하고 필요에 따라 효율적으로 gather하고 scatter합니다. **순수한 data parallelism과 비교하여, FSDP는 디바이스당 메모리 사용량을 크게 줄이고 역방향 패스 FLOP를 절약하며, 오버헤드는 매우 최소화합니다.**

{% include figure.liquid path="assets/img/fsdp.png" class="img-fluid" caption="<b>Figure:</b> FSDP는 Win의 contracting dimension과 Wout의 output dimension을 data dimension을 따라 샤딩합니다. 이는 메모리를 줄이지만 (Section 3에서) matmul을 수행하기 전에 W에 대한 weight을 gather해야 합니다. activation(왼쪽)은 <it>contracting dimension을 따라 샤딩되지 않는다</it>는 점에 주목하세요. 이것이 우리로 하여금 gather해야 하게 만듭니다. <b>weight optimizer state도 마찬가지로 contracting dimension을 따라 샤딩됩니다.</b>" %}

([Section 3](../sharding-kr)에서) AllReduce가 AllGather와 ReduceScatter로 분해될 수 있다는 것을 기억할 것입니다. 이는 표준 data parallelism에 대한 전체 gradient AllReduce를 하는 대신, 칩에 걸쳐 weight과 optimizer state를 샤딩하고, 순방향 패스 중에 각 layer에서 이들을 AllGather하고, 역방향 패스 중에 weight에 걸쳐 ReduceScatter를 추가 비용 없이 할 수 있다는 뜻입니다.

{% details FSDP의 전체 알고리즘입니다. %}

<div markdown=1 class="algorithm">

**Fully-Sharded Data Parallelism (FSDP):**

**순방향 패스:** Loss[B<sub>X</sub>]를 계산해야 함

1.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*critical path에 있지 않음, 이전 layer 중에 수행 가능*)
2.  Tmp[B<sub>X</sub>, F] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F] (*이제 W<sub>in</sub>[D, F]를 버릴 수 있음*)
3.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*critical path에 있지 않음, 이전 layer 중에 수행 가능*)
4.  Out[B<sub>X</sub>, D] = Tmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>out</sub>[F, D]
5.  Loss[B<sub>X</sub>] = ...

**역방향 패스:** dW<sub>out</sub>[F, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F]를 계산해야 함

1.  dOut[B<sub>X</sub>, D] = ...
2.  dW<sub>out</sub>[F, D] {U<sub>X</sub>} = Tmp[B<sub>X</sub>, F] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
3.  dW<sub>out</sub>[F, D<sub>X</sub>] = **ReduceScatter**(dW<sub>out</sub>[F, D] {U<sub>X</sub>}) (*critical path에 있지 않음, 비동기적으로 수행 가능*)
4.  W<sub>out</sub>[F, D] = **AllGather**(W<sub>out</sub>[F, D<sub>X</sub>]) (*미리 수행 가능*)
5.  dTmp[B<sub>X</sub>, F] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F, D] *(여기서 W<sub>out</sub>[F, D]를 버릴 수 있음)*
6.  dW<sub>in</sub>[D,F] {U<sub>X</sub>} = dTmp[B<sub>X</sub>, F] \*<sub>B</sub> In[B<sub>X</sub>, D]
7.  dW<sub>in</sub>[D<sub>X</sub>, F] = **ReduceScatter**(dW<sub>in</sub>[D, F] {U<sub>X</sub>}) *(critical path에 있지 않음, 비동기적으로 수행 가능)*
8.  W<sub>in</sub>[D, F] = **AllGather**(W<sub>in</sub>[D<sub>X</sub>, F]) (*미리 수행 가능*)
9.  dIn[B<sub>X</sub>, D] = dTmp[B<sub>X</sub>, F] \*<sub>F</sub> W<sub>in</sub>[D, F] (*이전 layer에 필요*) (여기서 W<sub>in</sub>[D, F]를 버릴 수 있음)

</div>

{% enddetails %}

이는 "ZeRO Sharding"라고도 불리며, "ZeRo Overhead sharding"에서 온 것으로 불필요한 계산을 수행하거나 불필요한 state를 저장하지 않기 때문입니다. ZeRO-{1,2,3}는 optimizer state, gradient, weight을 이런 방식으로 각각 샤딩하는 것을 나타내는 데 사용됩니다. 모두 동일한 통신 비용을 가지므로<d-footnote>기술적으로, FSDP는 순수한 DP가 갖지 않는 순방향 패스에 통신을 추가하지만, 이는 역방향 패스와 같은 비율이므로 comm roofline에 영향을 주지 않아야 합니다. 여기서 핵심은 ZeRO-3가 역방향 패스 AllReduce를 AllGather와 ReduceScatter로 바꾸는데, 이들은 동일한 총 comm 볼륨을 가진다는 것입니다.</d-footnote>, 기본적으로 항상 ZeRO-3 샤딩을 할 수 있으며, 이는 매개변수, gradient, optimizer state를 디바이스 세트에 걸쳐 샤딩합니다.

**왜 이렇게 하는가?** 표준 data parallelism은 많은 중복 작업을 포함합니다. 각 TPU는 전체 gradient를 AllReduce하고, 전체 optimizer state를 업데이트하고(모든 TPU에서 동일한 작업), 매개변수를 업데이트합니다(다시, 완전히 중복). ZeRO 샤딩(gradient/optimizer state 샤딩)의 경우, AllReduce 대신 gradient를 ReduceScatter하고, optimizer state의 샤드만 업데이트하고, 매개변수의 샤드를 업데이트한 다음, 순방향 패스에 필요에 따라 매개변수를 AllGather할 수 있습니다.

**언제 통신이 병목이 되는가?** 역방향 패스의 각 AllReduce가 AllGather + ReduceScatter가 되었으므로, 상대적 FLOP와 comm 비용은 순수한 data parallelism과 정확히 동일합니다. AllReduce가 AllGather와 ReduceScatter로 구현되며, 각각 절반의 비용을 가진다는 것을 기억하세요. 역방향 패스와 동일한 FLOP-to-comm 비율을 가지므로 여기서는 순방향 패스를 모델링합니다:

$$\begin{aligned}
T_{math} &= \frac{2 \cdot 2 \cdot B \cdot D \cdot F}{X \cdot C} \\
T_{comm} &= \frac{2 \cdot 2 \cdot D \cdot F}{W_\text{ici}} \\
T &\approx \max\left(\frac{4 \cdot B \cdot D \cdot F}{X \cdot C}, \frac{4 \cdot D \cdot F}{W_\text{ici}}\right) \\
T &\approx 4 \cdot D \cdot F \cdot \max\left(\frac{B}{X \cdot C}, \frac{1}{W_\text{ici}}\right)
\end{aligned}$$

따라서 순수한 data-parallelism과 마찬가지로, 디바이스당 배치 크기 $B/X$가 "ICI operational intensity" $C/W_\text{ici}$ (v5p의 경우 `4.59e14 / 1.8e11 = 2550`)를 초과할 때 $$B / X > C / W_\text{ici}$$에서 compute bound가 됩니다. 이는 우리에게 좋은 소식인데, 디바이스당 배치 크기가 순수한 data-parallelism에 대해 compute-bound가 될 만큼 충분히 크다면, compute-bound 영역을 벗어날 걱정 없이 단순히 FSDP로 업그레이드하여 매개변수와 optimizer state 메모리를 엄청나게 절약할 수 있기 때문입니다! 순방향 패스에 통신을 추가해야 했지만, 이 비용은 순방향 패스 FLOP와 겹치므로 중요하지 않습니다.

<p markdown=1 class="takeaway">**핵심 요점:** FSDP와 순수한 data parallelism 모두 TPUv5에서 디바이스당 배치 크기가 $2550 / n_\text{axes}$보다 작을 때 대역폭 바운드가 됩니다.</p>

예를 들어, DeepSeek-V2 (최근 훈련 배치 크기 정보를 공개한 몇 안 되는 강력한 모델 중 하나)는 약 40M 토큰의 배치 크기를 사용했습니다. **이는 대역폭 한계에 도달하기 전에 대략 47,000개의 칩, 또는 약 5개의 TPUv5 pod까지 스케일할 수 있게 해줍니다.**

약 `6.3e24 (15e12 * 70e9 * 6)` FLOP로 훈련된 LLaMA-3 70B의 경우, 16M 토큰 배치를 대략 `16e6 / (2550 / 3) = 18,823`개의 칩(대략 8960개 칩의 2개 pod)에 분할할 수 있으며, 각각 `4.59e14` FLOP를 50% 피크 FLOP 활용률(종종 MFU라고 함)로 실행하여 **약 17일 만에 훈련**할 수 있습니다. 나쁘지 않네요! 하지만 더 잘할 수 있는 방법을 탐색해 보겠습니다.

<p markdown=1 class="takeaway">**Critical batch size에 대한 참고**: 다소 직관적이지 않게, 총 배치 크기가 줄어들수록(칩 수는 고정) 더 통신 병목이 됩니다. Data parallelism과 FSDP는 배치 크기를 계속 늘릴 수 있는 한 임의로 많은 칩까지 스케일할 수 있습니다! 하지만 실제로는 배치 크기가 증가하면 gradient가 거의 노이즈가 없게 되어 훈련에서 수익 감소를 보는 경향이 있습니다. 때로는 훈련 불안정성도 봅니다. 따라서 "무제한 계산 영역"에서 최적의 샤딩 방식을 찾는 게임은 종종 스케일링 법칙에 의해 결정된 고정 배치 크기와 알려진(큰) 칩 수에서 시작하여, 그렇게 많은 칩에 그 작은 배치 크기를 맞출 수 있는 분할을 찾는 것을 목표로 합니다.</p>

### Tensor Parallelism

**문법:** $$\text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y]$$ (결국 FSDP와 결합하기 위해 $$Y$$를 사용)

fully-sharded data-parallel AllReduce에서는 칩에 걸쳐 weight을 이동합니다. 모델의 feedforward dimension을 샤딩하고 layer 중에 activation을 이동할 수도 있습니다 — 이를 "1D model parallelism" 또는 Megatron sharding<d-cite key="megatron"></d-cite>라고 합니다. 이는 pod당 더 작은 효율적인 배치 크기를 가능하게 할 수 있습니다. 아래 그림은 이런 방식으로 샤딩된 단일 matrix의 예를 보여줍니다:

{% include figure.liquid path="assets/img/model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> 기본적인 tensor parallelism의 예. Y에 대해서만 activation을 샤딩하므로(X에 대해 샤딩하는 FSDP와 달리), X에 대해 activation을 복제합니다. 표준 문법을 사용하면, <b>A</b>[B, D<sub>Y</sub>] * <b>B</b>[D, F<sub>Y</sub>] -> <b>C</b>[B, F<sub>Y</sub>]입니다. contracting dimension 중 하나에 대해서만 샤딩하므로, 일반적으로 matmul 전에 activation <b>A</b>를 AllGather합니다." %}

언급한 바와 같이, **In\[B, D<sub>Y</sub>\] \*<sub>D</sub> W<sub>in</sub>\[D, F<sub>Y</sub>\] \*<sub>F</sub> W<sub>out</sub>\[F<sub>Y</sub>, D\] \-\> Out\[B, D<sub>Y</sub>\]는 첫 번째 matmul 전에 activation을 gather해야 함을 의미합니다. 이는 activation이 weight보다 작을 때 ZeRO 샤딩보다 저렴합니다.** 이는 일반적으로 어느 정도의 ZeRO 샤딩이 추가된 경우에만 해당합니다(gather의 크기를 줄임). 이것이 ZeRO 샤딩과 model parallelism을 혼합하는 경향이 있는 이유 중 하나입니다.

{% details Tensor parallelism의 알고리즘입니다! %}

<div markdown=1 class="algorithm">

**Tensor Parallelism:**

**순방향 패스:** Loss[B]를 계산해야 함

1.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(critical path에 있음)*
2.  Tmp[B, F<sub>Y</sub>] = In[B, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(contracting을 따라 샤딩되지 않음, 따라서 comm 없음)*
3.  Out[B, D] {U<sub>Y</sub>} = Tmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
4.  Out[B, D<sub>Y</sub>] = **ReduceScatter**(Out[B, D] {U<sub>Y</sub>}) *(critical path에 있음)*
5.  Loss[B] = ...

**역방향 패스:** dW<sub>out</sub>[F<sub>Y</sub>, D], dW<sub>in</sub>[D, F<sub>Y</sub>]를 계산해야 함

1.  dOut[B, D<sub>Y</sub>] = ...
2.  dOut[B, D] = **AllGather**(dOut[B, D<sub>Y</sub>]) *(critical path에 있음)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] = Tmp[B, F<sub>Y</sub>] \*<sub>B</sub> dOut[B, D]
4.  dTmp[B, F<sub>Y</sub>] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(여기서 dOut[B, D]를 버릴 수 있음)*
5.  In[B, D] = **AllGather**(In[B, D<sub>Y</sub>]) *(순방향 패스의 (1)과 공유하여 건너뛸 수 있음)*
6.  dW<sub>in</sub>[D, F<sub>Y</sub>] = dTmp[B, F<sub>Y</sub>] \*<sub>B</sub> In[B, D]
7.  dIn[B, D] {U.Y} = dTmp[B, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(이전 layer에 필요)*
8.  dIn[B, D<sub>Y</sub>] = **ReduceScatter**(dIn[B, D] {U.Y}) *(critical path에 있음)*

</div>

{% enddetails %}

Tensor parallelism의 좋은 점 중 하나는 Transformer 순방향 패스의 두 matrix와 잘 상호작용한다는 것입니다. 순진하게는 두 matrix 각각 후에 AllReduce를 할 것입니다. 하지만 여기서는 먼저 **In[B, D<sub>Y</sub>] \* W<sub>in</sub>[D, F<sub>Y</sub>] -> Tmp[B, F<sub>Y</sub>]**를 하고 그 다음 **Tmp[B, F<sub>Y</sub>] \* W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B, D<sub>Y</sub>]**를 합니다. 이는 AllReduce를 하는 대신 처음에 **In**을 AllGather하고 끝에 **Out**을 ReduceScatter한다는 뜻입니다.

**이 비용은 얼마나 될까요?** 순방향 패스만 모델링하겠습니다 - 역방향 패스는 여기서 각 연산의 transpose일 뿐입니다. 1D model parallelism에서는 첫 번째 matmul 전에 activation을 AllGather하고, 두 번째 후에 ReduceScatter하여, 한 번에 두 바이트씩(bf16) 보냅니다. 언제 통신이 병목이 되는지 알아보겠습니다.

$$\begin{align}
T_{math} & = \frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} \\
T_{comms} & =
\frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\\
\textnormal{T} & \approx \max \left(\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C}, \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}\right)
\end{align}$$

계산 비용이 comm 비용보다 크기를 원한다는 점에 주목하면:

$$\begin{align}
\frac{4 \cdot B \cdot D \cdot F}{Y \cdot C} > \frac{2 \cdot 2 \cdot (B \cdot D)}{W_\text{ici}}
\end{align}$$

$$\begin{align}
\frac{F}{Y \cdot C} > \frac{1}{W_\text{ici}}
\end{align}$$

$$\begin{align}
F > Y \cdot \frac{C}{W_\text{ici}}
\end{align}$$

따라서 예를 들어, TPUv5p의 경우 $$C / W_{ici} = 2550$$이므로(bf16에서), $$Y < F / 2550$$까지만 tensor parallelism을 할 수 있습니다. 여러 ICI axis가 있을 때, $$T_\text{comms}$$가 $n_\text{axes}$만큼 줄어들므로 $$Y < n_\text{axes} * F / 2550$$을 얻습니다.

<p markdown=1 class="takeaway">**핵심 요점**: model parallelism은 $$Y > n_\text{axes} * F / 2550$$일 때 통신 바운드가 됩니다. 대부분의 모델에서 이는 8-16-way model parallelism 사이입니다.</p>

**이는 계산의 precision에 의존하지 않는다**는 점에 주목하세요. 예를 들어 int8의 경우, TPUv5p에서 $$C_\text{int8} / W_{ici}$$는 $$2550$$ 대신 $$5100$$이지만 comm 볼륨도 절반이므로, 두 2의 인수가 상쇄됩니다.

**몇 가지 예를 생각해 보겠습니다:**

* $$D = 8192,$$ $$F \approx 30,000$$인 LLaMA 3-70B를 가진 TPUv4p에서, 8-way model parallelism을 편안하게 할 수 있지만 16 way model parallelism에서는 통신 바운드가 될 것입니다. 8 way model 샤딩에 필요한 F는 20k입니다.

* Gemma 7B의 경우, $$F \approx 50k$$이므로 19-way model parallelism에서 통신 바운드가 됩니다. 즉, 16-way를 할 수 있고 여전히 좋은 성능을 볼 가능성이 높습니다.

### 혼합 FSDP와 Tensor Parallelism

**문법:** $$\text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]$$

FSDP와 tensor parallelism의 좋은 점은 결합될 수 있다는 것입니다. **W<sub>in</sub>**과 **W<sub>out</sub>**을 두 axis 모두를 따라 샤딩함으로써 메모리와 계산을 모두 절약합니다. B를 X를 따라 샤딩하므로 model-parallel AllGather의 크기를 줄이고, F를 Y를 따라 샤딩하므로 FSDP의 통신 오버헤드를 줄입니다. 이는 둘의 조합이 위에서 본 것보다 훨씬 낮은 효과적인 배치 크기에 도달할 수 있음을 의미합니다.

{% include figure.liquid path="assets/img/mixed-fsdp-model-parallelism.png" class="img-fluid" caption="<b>Figure:</b> FSDP와 tensor parallelism을 결합한 다이어그램. 다른 경우와 달리, 모델 매개변수의 중복이 없습니다." %}

{% details 혼합 FSDP + tensor parallelism의 전체 알고리즘입니다. 많은 통신이 있지만, 모든 AllGather와 ReduceScatter는 배치 샤딩된 activation과 텐서 샤딩된 weight을 훨씬 더 많이 했기 때문에 더 작습니다! %}

<div markdown=1 class="algorithm">

**순방향 패스:** Loss[B]를 계산해야 함

1.  In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있음)*
2.  W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(미리 수행 가능)*
3.  Tmp[B<sub>X</sub>, F<sub>Y</sub>] = In[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>]
4.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(미리 수행 가능)*
5.  Out[B<sub>X</sub>, D] {U.Y} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D]
6.  Out[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(Out[B<sub>X</sub>, D] {U.Y}) *(critical path에 있음)*
7.  Loss[B<sub>X</sub>] = ...

**역방향 패스:** dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>], dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]를 계산해야 함

1.  dOut[B<sub>X</sub>, D<sub>Y</sub>] = ...
2.  dOut[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(dOut[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있음)*
3.  dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X} = Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> dOut[B<sub>X</sub>, D]
4.  dW<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>out</sub>[F<sub>Y</sub>, D] {U.X})
5.  W<sub>out</sub>[F<sub>Y</sub>, D] = **AllGather**<sub>X</sub>(W<sub>out</sub>[F<sub>Y</sub>, D<sub>X</sub>]) *(미리 수행 가능)*
6.  dTmp[B<sub>X</sub>, F<sub>Y</sub>] = dOut[B<sub>X</sub>, D] \*<sub>D</sub> W<sub>out</sub>[F<sub>Y</sub>, D] *(여기서 dOut[B, D]를 버릴 수 있음)*
7. In[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(In[B<sub>X</sub>, D<sub>Y</sub>]) *(critical path에 있지 않음 + 이전 layer의 (2)와 공유 가능)*
8.  dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>B</sub> In[B<sub>X</sub>, D]
9.  dW<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>] = **ReduceScatter**<sub>X</sub>(dW<sub>in</sub>[D, F<sub>Y</sub>] {U.X})
10. W<sub>in</sub>[D, F<sub>Y</sub>] = **AllGather**<sub>X</sub>(W<sub>in</sub>[D<sub>X</sub>, F<sub>Y</sub>]) *(미리 수행 가능)*
11. dIn[B<sub>X</sub>, D] {U.Y} = dTmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W<sub>in</sub>[D, F<sub>Y</sub>] *(이전 layer에 필요)*
12. dIn[B<sub>X</sub>, D<sub>Y</sub>] = **ReduceScatter**<sub>Y</sub>(dIn[B<sub>X</sub>, D] {U.Y}) *(critical path에 있음)*

</div>

{% enddetails %}

**FSDP와 MP의 적절한 조합은 무엇인가?** 간단하지만 핵심적인 격언은 FSDP는 weight을 이동시키고 model parallelism은 activation을 이동시킨다는 것입니다. 즉, 배치 크기가 줄어들수록 (특히 더 많은 data parallelism을 할수록) 샤드당 activation이 작아지므로 model parallelism이 저렴해집니다.

* Model parallelism은 $$\mathbf{AllGather}_Y([B_X, D_Y])$$를 수행하며 이는 $$X$$가 증가하면서 줄어듭니다.
* FSDP는 $$\mathbf{AllGather}_X([D_X, F_Y])$$를 수행하며 이는 $$Y$$가 증가하면서 줄어듭니다.

따라서 둘을 결합함으로써 복제본당 최소 배치 크기를 더욱 줄일 수 있습니다. 위와 같은 방식으로 최적의 FSDP와 MP 양을 계산할 수 있습니다:

$$X$$를 FSDP에 전용된 칩 수, $$Y$$를 tensor parallelism에 전용된 칩 수라고 하겠습니다. $$N$$을 $$N=XY$$인 슬라이스의 총 칩 수라고 하겠습니다. $$M_X$$와 $$M_Y$$를 각각 FSDP와 MP를 수행하는 mesh axis의 수라고 하겠습니다(이들은 대략 3의 합이어야 합니다). FLOP당 통신이 가장 많으므로 순방향 패스만 순수하게 모델링하겠습니다. 그러면 위 알고리즘에서 comm을 더하면:

$$T_\text{FSDP comms}(B, X, Y) = \frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X}$$

$$T_\text{MP comms}(B, X, Y) = \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y}$$

마찬가지로 총 FLOP 시간은

$$T_\text{math} = \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}.$$

분석을 단순화하기 위해 두 가지 단순화를 합니다: 첫째, $X$와 $Y$가 (양수이고 $XY=N$을 만족하는 한) 비정수 값을 가질 수 있도록 허용합니다; 둘째, $X$와 $Y$ axis에서 comm을 겹치지 않는다고 가정합니다. 두 번째 가정 하에서, 총 comm 시간은

$$T_\text{comms} = T_\text{FSDP comms} + T_\text{MP comms}.$$


compute-bound가 되는 조건을 묻기 전에, 총 통신을 최소화하는 $X$와 $Y$의 최적값을 찾아보겠습니다. FLOP가 $X$와 $Y$와 독립적이므로, 최적 설정은 단순히 comm을 최소화하는 것들입니다. 이를 위해, 위의 $T_\text{comms}$를 $X$와 $Y$ 대신 $X$와 $N$(고정된 시스템의 칩 수)로 표현해 보겠습니다:

$$T_\text{comms} (X) = \frac{F \cdot X}{N \cdot M_X} + \frac{B}{X \cdot M_Y}$$

이 식을 $X$에 대해 미분하고 도함수를 0으로 설정하면 최적값 $X_{opt}$를 얻습니다:

$$\begin{align*}
\frac{d}{dX} T_\text{comms} (X_{opt}) = \frac{F}{N \cdot M_X} - \frac{B}{X_{opt}^2 \cdot M_Y} \rightarrow \\
X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}
\end{align*}$$

이는 매우 유용합니다! 이는 주어진 $B$, $F$, $N$에 대해 어느 정도의 FSDP가 최적인지 알려줍니다. 규모를 알아보겠습니다. 현실적인 값들, 즉 $N = 64$ (4x4x4 칩 배열에 해당), $B=48,000$, $F=32,768$를 넣으면 대략 $X\approx 13.9$가 나옵니다. 따라서 $X$를 16, $Y$를 4로 선택할 것이며, 이는 계산된 최적값에 가깝습니다.

<p markdown=1 class="takeaway">**핵심 요점:** 일반적으로 훈련 중 최적의 FSDP 양은 $$X_{opt} = \sqrt{\frac{B}{F} \frac{M_X}{M_Y} N}$$입니다. </p>

이제 모든 병렬화 전략에 대해 물어온 질문으로 돌아가겠습니다: **어떤 조건에서 compute-bound가 될까요?** FLOP와 comm을 겹칠 수 있으므로, 다음 조건일 때 compute-bound입니다:

$$T_\text{FSDP comms} + T_\text{MP comms} < T_\text{math}$$

이는 다음을 제공합니다:

$$\frac{2\cdot 2\cdot D \cdot F}{Y \cdot W_\text{ici} \cdot M_X} + \frac{2 \cdot 2 \cdot B \cdot D}{X \cdot W_\text{ici} \cdot M_Y} < \frac{2\cdot 2 \cdot B \cdot D \cdot F}{N \cdot C}$$

$\alpha \equiv C / W_\text{ici}$를 ICI arithmetic intensity라고 하면, 단순화할 수 있습니다:

$$\frac{F}{Y \cdot M_X} + \frac{B}{X \cdot M_Y} < \frac{B \cdot F}{N \cdot \alpha}$$

계산된 $X_{opt}$를 위 방정식에 넣으면 ($Y_{opt} = N/X_{opt}$임을 주목) 배치 크기 $B$에 대한 다음 조건이 나옵니다:

$$ \sqrt{\frac{4 \cdot B\cdot F}{M_X \cdot M_Y \cdot N}} < \frac{B \cdot F}{N \cdot \alpha},$$

여기서 좌변은 통신 시간에 비례하고 우변은 계산 시간에 비례합니다. 계산 시간이 (병렬화와 관계없이) 배치 크기에 선형으로 스케일되는 반면, 통신 시간은 배치 크기의 제곱근으로 스케일됩니다. 따라서 계산 대 통신 시간의 비율도 배치 크기의 제곱으로 스케일됩니다:

$$ \frac{T_\text{math}}{T_\text{comms}} = \frac{\sqrt{BF}\sqrt{M_X M_Y}}{2\alpha \sqrt{N}}. $$

compute bound가 되도록 이 비율이 1보다 큰지 확인하려면 다음이 필요합니다:

$$ \frac{B}{N} > \frac{4\alpha^2}{M_X M_Y F}$$

이 관계의 대안적 도출은 부록 C를 참조하세요. 대략적인 숫자를 얻기 위해, 다시 $F=32,768$, $\alpha=2550$, $M_X M_Y=2$ (3D mesh의 경우 그래야 함)를 넣으면 대략 $B/N > 400$이 나옵니다. 이는 3D mesh를 가정할 때 $B/N$이 약 $850$을 초과해야 compute bound가 되는 순수한 data parallel (또는 FSDP) 경우와 비교하여 대략 2배의 이점을 얻습니다.

<p markdown=1 class="takeaway">**핵심 요점:** tensor parallelism과 FSDP를 결합하면 $B/N$을 $$2 \cdot 2550^2 / F$$까지 떨어뜨릴 수 있습니다. 이는 칩당 최소 400 배치를 처리할 수 있게 해주며, 이는 FSDP만으로 달성할 수 있는 것보다 대략 2배 작습니다.</p>

아래에서는 대표적인 4x4x4 칩 배열에서 혼합 FSDP + MP에 대한 FLOP 대 comm 시간 비율을 model parallelism만과 data parallelism (FSDP)만과 비교하여 플롯했습니다. 순수한 FSDP parallelism이 매우 큰 배치 크기에서 지배적인 반면, 칩 수당 배치 크기가 대략 400과 850 사이인 영역에서는 compute-bound가 되기 위해 혼합 FSDP + MP 전략이 필요합니다.

{% include figure.liquid path="assets/img/mixed-fsdp-comms-2.png" class="img-fluid" caption="<b>Figure:</b> F=30k인 TPUv5p 4x4x4 슬라이스에서 최적 혼합 FSDP/MP의 FLOP 대 comm 시간 비율. 예상대로, model parallelism은 배치 크기와 고정 비율을 가지고; 이상적인 혼합 FSDP + MP는 $\sqrt{B}$로 스케일되고, FSDP는 $B$로 스케일됩니다. 그러나 중간 배치 크기 영역에서는 FSDP + MP만이 1보다 큰 비율을 달성합니다."%}

다음은 다른 샤딩 방식에 대한 배치 크기의 함수로서 FLOP와 comm 시간을 보여주는 TPU v5p 16x16x16의 또 다른 예입니다.

{% include figure.liquid path="assets/img/comms-flops-time.png" class="img-fluid" caption="<b>Figure:</b> 다른 병렬화 방식의 통신에 소요되는 시간. 검은 점선은 matrix multiplication FLOP에 소요되는 시간이므로, 이 선 위의 모든 곡선은 comm-bound입니다. 모든 전략이 배치 크기 1.5e6 아래에서 comm-bound가 되는 것을 확인할 수 있으며, 이는 예상된 4096 * 2 * 2550^2 / (8192 * 4) = 1.6e6과 일치합니다." %}

검은 곡선은 모델 FLOP에 소요되는 시간이므로, 이보다 낮은 모든 comm 비용보다 낮은 배치 크기는 엄격히 comm bound입니다. 검은 곡선이 예측대로 약 `1.6e10`에서 녹색 곡선과 교차하는 것을 볼 수 있습니다.

확대하면, FSDP에 두 축을 할당하고 광학 스위치를 사용하여 토폴로지를 model 샤딩을 위한 8-long axis를 갖도록 재구성하는 것이 1M과 6M 배치 크기 사이에서 가장 낮은 통신 볼륨을 제공하는 반면, 순수한 FSDP 조합이 6M과 100M 사이에서 최고라는 것을 볼 수 있습니다. 이는 위의 계산과 일치합니다!

{% include figure.liquid path="assets/img/comms-flops-time-zoom.png" class="img-fluid" %}

다음은 다른 배치 크기에 대한 총 계산 시간과 통신 시간을 보여주는 대화형 애니메이션입니다:

<div class="l-page">
  <iframe src="{{ 'assets/plotly/training-roofline.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

이는 일반적으로 위와 일치함을 볼 수 있습니다 (FSDP=256, MP=16 주변에서 최소), 각각에 대한 axis 수의 약간의 차이에 대한 약간의 변동 요인을 더하거나 뺀 것입니다.

### Pipelining

이전 섹션에서 pipelining에 대해 전혀 이야기하지 않았다는 것을 아마 눈치챘을 것입니다. Pipelining은 TPU에서는 다소 덜 필수적인 GPU 병렬화의 지배적인 전략입니다. 간략히, pipelined 훈련은 모델의 layer를 여러 디바이스에 걸쳐 분할하고 순방향 및 역방향 패스 중에 파이프라인 스테이지 간에 activation을 전달하는 것을 포함합니다. 알고리즘은 다음과 같습니다:

1. FSDP 및 tensor parallelism과 함께 pipelining을 위해 layer dimension에 걸쳐 샤딩된 weight으로 TPU 0에서 데이터를 초기화합니다 ($W_\text{in}[L_Z, D_X, F_Y]$).
2. TPU 0에서 첫 번째 layer를 수행한 다음, 결과 activation을 TPU 1에 복사하고, 마지막 TPU에 도달할 때까지 반복합니다.
3. loss function과 그 도함수 $\partial L / \partial x_L$을 계산합니다.
4. 마지막 파이프라인 스테이지의 경우, 도함수 $\partial L / \partial W_L$과 $\partial L / \partial x_{L-1}$을 계산한 다음, $\partial L / \partial x_{L-1}$을 이전 파이프라인 스테이지에 복사하고 TPU 0에 도달할 때까지 반복합니다.

{% details 다음은 (작동하는) Python 의사 코드입니다 %}

이 의사 코드는 Cloud TPU VM에서 실행되어야 합니다. 매우 효율적이거나 현실적이지는 않지만, 디바이스 간에 데이터가 어떻게 전파되는지에 대한 감각을 제공합니다.

```python
batch_size = 32
d_model = 128
d_ff = 4 * d_model

num_layers = len(jax.devices())

key = jax.random.PRNGKey(0)

# 각 layer가 단일 matmul이라고 가정합니다.
x = jax.random.normal(key, (batch_size, d_model))
weights = jax.random.normal(key, (num_layers, d_model, d_model)) 

def layer_fn(x, weight):
  return x @ weight

# num_layers == num_pipeline_stages라고 가정합니다
intermediates = [x]
for i in range(num_layers):
  x = layer_fn(x, weights[i])
  intermediates.append(x)

  if i != num_layers - 1:
    x = jax.device_put(x, jax.devices()[i+1])

def loss_fn(batch):
  return jnp.mean(batch ** 2)  # 가짜 loss function을 만듭니다

loss, dx = jax.value_and_grad(loss_fn)(x)

for i in range(0, num_layers, -1):
  _, f_vjp = jax.vjp(layer_fn, intermediates[i + 1], weights[i])
  dx, dw = f_vjp(dx)  # jvp dx @ J(L)(x[i], W[i])를 계산합니다
  weights[i] = weights[i] - 0.01 * dw  # weight을 업데이트합니다

  if i != 0:
    dx = jax.device_put(dx, jax.devices()[i-1])
```

{% enddetails %}

**왜 이것이 좋은 아이디어인가?** Pipelining은 여러 이유로 훌륭합니다: 파이프라인 스테이지 간 통신 비용이 낮아서, 낮은 대역폭 interconnect로도 매우 큰 모델을 훈련할 수 있습니다. 이는 GPU가 TPU처럼 ICI로 조밀하게 연결되지 않기 때문에 GPU에서 종종 매우 유용합니다.

**왜 이것이 어렵거나/성가신가?** 위의 의사 코드에서 TPU 0이 거의 항상 유휴 상태라는 것을 눈치챘을 것입니다! 파이프라인의 맨 처음과 마지막 단계에서만 작업을 하고 있습니다. 이 유휴 기간을 파이프라인 버블이라고 하며 다루기 매우 성가십니다. 일반적으로 먼저 microbatching으로 이를 완화하려고 시도하는데, 이는 여러 작은 배치를 파이프라인을 통해 보내서 TPU 0이 최소한 전체 스텝 시간의 더 큰 부분 동안 활용되도록 유지합니다.

두 번째 접근법은 순방향 matmul $W_i @ x_i$, 역방향 $dx$ matmul $W_i @ \partial L / \partial x_{i+1}$, 그리고 $dW$ matmul $\partial L / \partial x_{i+1} @ x_i$를 신중하게 겹치는 것입니다. 각각이 일부 FLOP를 필요로 하므로, 이들을 겹쳐서 버블을 완전히 숨길 수 있습니다. 다음은 최근 DeepSeek v3 논문<d-cite key="DeepSeek3"></d-cite>에서 "버블 없는" 파이프라인 스케줄을 보여주는 플롯입니다:

{% include figure.liquid path="assets/img/deepseek-pipeline.png" class="img-fluid" caption="<b>Figure:</b> DeepSeek v3 파이프라인 스케줄 (<a href=\"https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf\">최근 논문</a>에서). 주황색은 순방향 matmul, 녹색은 dL/dx matmul, 파란색은 dL/dW matmul입니다. 역방향 dL/dx multiplication을 우선시함으로써, FLOP를 \"좌초\"시키는 것을 피할 수 있습니다." %}

TPU (더 큰 상호연결된 pod를 가짐)에는 덜 중요하므로, 이에 대해 깊이 들어가지는 않겠지만, 핵심 pipelining 병목을 이해하는 것은 좋은 연습입니다.

### Pod 간 스케일링

한 걸음 물러서서 구체적인 예를 보겠습니다. TPU v5p에서 LLaMA-3 70B를 훈련한다고 해보겠습니다. LLaMA-3 70B는 $$F\approx 30,000$$을 가집니다. 위 섹션에서 다음을 알고 있습니다:

* $$Y > n_\text{axes} * F / 2550 \approxeq n_\text{axes} * 11$$ 이상의 model parallelism을 할 때 ICI bound가 됩니다.
* 순수한 FSDP는 $$\text{batch size} < 2550 / n_\text{axes}$$일 때 ICI bound가 됩니다. 여기서 BS=2M으로 훈련하고 싶다면, 최대 약 2400개의 칩을 사용할 수 있으며, 이는 TPU v5p pod의 약 1/4입니다.
* 혼합 FSDP + model parallelism은 $$\text{batch size} < 2 \cdot 2550^2 / 30,000 = 432$$일 때 ICI bound가 되므로, 이는 대략 9k 칩까지 스케일할 수 있게 해줍니다! 하지만 TPU v5p pod의 최대 크기는 8k 칩이고, 그 이상으로는 더 낮은 대역폭의 data-center networking (DCN)으로 스케일해야 합니다.

따라서 이는 BS=3.5M으로 단일 pod에 맞추는 좋은 레시피를 제공합니다. 위 방정식을 사용하면 대략 X (FSDP) = 1024, Y (MP) = 8이 나옵니다. 모델이 더 크다면, model 샤딩을 16으로 확장할 여지가 있을 것입니다. 해당 pod에서 BS=1.5M까지 배치 크기를 떨어뜨리고 여전히 compute bound가 될 여지가 약간 있지만, 거기서 하한에 가깝습니다.

**하나의 pod보다 더 크게 가려면, DCN을 통해 스케일해야 합니다.** DCN은 더 낮은 대역폭을 가지므로, 일반적으로 많은 유용한 FSDP를 하기에는 너무 느립니다. 대신, DCN axis를 통해 순수한 data parallelism을 하고 pod 내에서 FSDP를 합니다. Data Center Network (DCN)이 견딜 수 있는지 계산해 보겠습니다.

DCN을 통한 순수한 data parallelism으로, 각 스텝 중에 weight과 optimizer state를 동기화해야 합니다 (모델이 역방향 패스를 완료할 때 AllReduce를 완료해야 함). 실제로 위의 순수한 data parallelism 섹션에서 수학을 빌려올 수 있으며, 이는 $\text{per pod batch size} < C_\text{pod} / W_\text{dcn}$일 때 comm bound가 된다고 알려줍니다. 여기서 RHS는 전체 pod의 총 계산과 총 대역폭입니다.

* 총 DCN ingress+egress 대역폭은 호스트당 2.5e10이고, 호스트당 4개의 칩이 있습니다. 슬라이스에 약 2000개의 호스트가 있으므로, 총 `5e13` 바이트의 대역폭입니다.
* 여기서 $$C_\text{pod}$$는 pod 크기 곱하기 칩당 계산이므로, `8k * 4.5e14 = 3.8e18` FLOP입니다.

이전과 마찬가지로, $T_\text{math} < T_\text{comms}$일 때 병목이 되며, 이는 $\text{per pod batch size} < C / W_\text{DCN} = 3.8e18 / 5e13 = 76,000$ (pod 수준 DCN operational intensity)일 때 발생합니다. LLaMA-3의 경우, pod당 배치 크기가 그것보다 훨씬 높으므로 문제가 되지 않을 것이지만, 더 작은 슬라이스 (예: v5e)에서 훈련한다면 문제가 될 수 있습니다.

<p markdown=1 class="takeaway">**핵심 요점:** 이는 pod에 걸쳐 상당히 임의로 스케일할 수 있음을 의미하므로, 예를 들어 8960개 칩의 10개 pod로 89,600개 칩에서 약 40M 토큰의 글로벌 배치 크기로 LLaMA-3 70B를 약 2일 만에 훈련할 수 있습니다.</p>

## TPU에서 LLM 훈련의 핵심 요점

* 병렬화를 늘리거나 배치 크기를 줄이는 것 모두 칩당 수행되는 계산량을 줄이기 때문에 통신 바운드가 되기 쉽게 만듭니다.

* 합리적인 컨텍스트 길이(~32k)까지는 Transformer를 MLP 블록의 스택으로 모델링하고 layer당 두/세 개의 주요 matmul을 어떻게 샤딩하는지에 따라 여러 병렬화 방식을 정의할 수 있습니다.

* 훈련 중에는 고려하는 4가지 주요 병렬화 방식이 있으며, 각각 고유한 대역폭과 계산 요구사항을 가집니다 (data parallelism, FSDP, model parallelism).

| **전략**                                 | **설명**                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Parallelism**                         | Activation은 배치 샤딩되고, 나머지는 모두 완전 복제되며, 역방향 패스 중에 gradient를 all-reduce합니다.                                                                      |
| **FSDP**                                     | Activation, weight, optimizer는 배치 샤딩되고, weight은 사용 직전에 gather되며, gradient는 reduce-scatter됩니다.                                                               |
| **Model Parallelism (aka Megatron, Tensor)** | Activation은 $$d_\text{model}$$을 따라 샤딩되고, weight은 $$d_{ff}$$를 따라 샤딩되며, activation은 W<sub>in</sub> 전에 gather되고, 결과는 W<sub>out</sub> 후에 reduce-scatter됩니다. |
| **Mixed FSDP + Model Parallelism**           | 위 둘 모두, FSDP가 model 샤딩된 weight을 gather합니다.                                                                                                           |

각 방법에 대한 "공식"은 다음과 같습니다:

$$\small
\begin{array}{cc}
\text{전략} & \text{공식}\\
\hline
\text{DP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D, F] \cdot_F W_\text{out}[F, D] \rightarrow \text{Out}[B_X, D] \\
\text{FSDP} & \text{In}[B_X, D] \cdot_D W_\text{in}[D_X, F] \cdot_F W_\text{out}[F, D_X] \rightarrow \text{Out}[B_X, D] \\
\text{MP} & \text{In}[B, D_Y] \cdot_D W_\text{in}[D, F_Y] \cdot_F W_\text{out}[F_Y, D] \rightarrow \text{Out}[B, D_Y] \\
\text{MP + FSDP}  & \text{In}[B_X, D_Y] \cdot_D W_\text{in}[D_X, F_Y] \cdot_F W_\text{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y] \\
\hline
\end{array}$$

* 이들 전략 각각은 디바이스당 계산과 comm에 기반하여 네트워크/통신 바운드가 되는 한계를 가집니다. 다음은 $$X$$가 FSDP이고 $$Y$$가 model parallelism이라고 가정한 layer당 계산과 comm입니다.

$$
\small
\begin{array}{ccc}
\text{전략} & \text{layer당 계산} & \text{layer당 comm} \\
& \text{(gating einsum 무시)} & \text{(바이트, 순방향 + 역방향 패스)}\\
\hline
\text{DP} & 4BDF/X + 8BDF/X & 0 + 8DF \\
\text{FSDP} & 4BDF/X + 8BDF/X & 4DF + 8DF \\
\text{MP} & 4BDF/Y + 8BDF/Y & 4BD + 4BD \\
\text{FSDP + MP} & 4BDF/(XY) + 8BDF/(XY) & (4BD/X + 4DF/Y) + (8BD/X + 8DF/Y) \\
\hline
\end{array}$$

* 순수한 data parallelism은 모델과 optimizer state가 매개변수 수의 10배 바이트를 사용하므로 거의 유용하지 않습니다. 이는 메모리에 수십억 개의 매개변수 이상을 거의 맞출 수 없음을 의미합니다.

* Data parallelism과 FSDP는 $$\text{batch size per shard} < C / W$$, 즉 네트워크의 arithmetic intensity일 때 comm bound가 됩니다. ICI의 경우 이는 2,550이고 DCN의 경우 75,000입니다. 이는 더 많은 병렬 axis로 증가시킬 수 있습니다.

* Model parallelism은 $$\lvert Y\rvert > F / 2550$$일 때 comm bound가 됩니다. **대부분의 모델에서 이는 8-16 way 정도입니다.** 이는 배치 크기와 독립적입니다.

* 혼합 FSDP + model parallelism은 배치 크기를 $$2 \cdot 2550^2 / F \approx 400$$까지 낮출 수 있게 해줍니다. 이는 어차피 HBM 대역폭 바운드가 되는 지점(~200)에 상당히 가깝습니다.

* Pod 간 data parallelism은 DCN-bound가 되기 전에 pod당 대략 75,000의 최소 배치 크기가 필요합니다.

* 기본적으로, 배치 크기가 크거나 모델이 작다면 상황이 간단합니다. data parallelism 또는 DCN에 걸친 FSDP \+ data parallelism을 할 수 있습니다. 중간 섹션이 흥미로운 곳입니다.

## 연습 문제

이 섹션의 기본 모델로 LLaMA-2 13B를 사용하겠습니다. 다음은 몇 가지 세부사항입니다:

| hyperparam              | value  |
| ----------------------- | ------ |
| n\_layers (L)           | 40     |
| d\_model (D)            | 5,120  |
| ffw\_multiplier (F / D) | 2.7    |
| n\_heads (N)            | 40     |
| n\_kv\_heads (K)        | 40     |
| d\_qkv (H)              | 128    |
| n\_embeddings (V)       | 32,000 |

**질문 1:** LLaMA-2 13B는 몇 개의 매개변수를 가지고 있나요 (우스꽝스럽다는 것을 알지만 계산해 보세요)? *참고로, [Transformer Math](../transformers-kr)에서와 같이, LLaMA-3는 3개의 큰 FFW matrix, 두 개의 up-projection과 하나의 down-projection을 가집니다. 이 섹션에서는 두 개의 "gating" einsum matrix를 무시했지만, 이들은 이 섹션에서 W<sub>in</sub>과 동일하게 동작합니다.*

{% details 답을 보려면 여기를 클릭하세요. %}

* FFW 매개변수: $$3LDF$$ = `8.5e9`
* Attention 매개변수: $$4DNHL$$ = `4.2e9`
* Vocabulary 매개변수: $$2VD$$ = `0.3e9`
* 총합: `8.5e9 + 4.2e9 + 0.39e9 = 13.1e9`, 예상대로!

{% enddetails %}

**질문 2:** BS=16M 토큰으로 훈련하고 Adam을 사용한다고 가정하겠습니다. 잠시 병렬화를 무시하고, 모델의 매개변수, optimizer state, activation에 의해 사용되는 총 메모리는 얼마인가요? *매개변수를 bf16으로, optimizer state를 fp32로 저장하고 layer당 세 번(세 개의 큰 matmul 후) activation을 체크포인트한다고 가정하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

매개변수(bf16)와 두 optimizer state(fp32, 1차 및 2차 moment accumulator)에 사용되는 총 메모리는 `(2 + 4 + 4) * 13e9 ~ 130GB`입니다. 처음 두 matmul 후의 activation은 $BF$ 모양이고 마지막 것 후에는 $BD$입니다 (위 Transformer 다이어그램에 따라), 따라서 bf16에 대한 총 메모리는 $2 \cdot L \cdot (BD + 2 * BF) = 2LB \cdot (D + 2F)$ 또는 `2 * 40 * 16e6 * 5,120 * (1 + 2 * 2.7) ~ 4.2e13 = 42TB`입니다. `B=16e16`이므로. 다른 모든 activation은 거의 무시할 수 있습니다.

{% enddetails %}

**질문 3:** 32k 시퀀스 길이와 TPUv5p 16x16x16 슬라이스에서 3M 토큰의 총 배치 크기로 훈련하고 싶다고 가정하겠습니다. 위와 같이 bfloat16 weight과 float32 optimizer를 사용한다고 가정하겠습니다.

1. 순수한 data parallelism을 사용할 수 있나요? 왜 그렇거나 그렇지 않나요?
2. 순수한 FSDP를 사용할 수 있나요? 왜 그렇거나 그렇지 않나요? 순수한 FSDP로, 디바이스당 얼마나 많은 메모리가 사용될까요 (3개의 큰 FFW matrix 후에만 gradient checkpointing을 한다고 가정)?
3. 혼합 FSDP + model parallelism을 사용할 수 있나요? 왜 그렇거나 그렇지 않나요? 그렇다면, $X$와 $Y$는 무엇이어야 할까요? 디바이스당 얼마나 많은 메모리가 저장될까요? roofline FLOP 추정만 사용하고 attention을 무시하면, 각 훈련 스텝은 얼마나 걸릴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

먼저, 몇 가지 숫자를 써보겠습니다. 32k 시퀀스 길이와 3M 배치 크기로, 시퀀스 배치 크기는 96입니다. TPU v5p 16x16x16 슬라이스에서, `393TB`의 HBM이 있습니다.

1. 각 칩에 매개변수와 optimizer state를 복제하므로 순수한 data parallelism을 사용할 수 없으며, 이는 이미 약 130GB (Q2에서)로 칩당 HBM(96GB)보다 많습니다.

2. 먼저 순전히 메모리를 살펴보겠습니다. Q2에서 BS=16M을 3M으로 바꾸면 `~7.86e12` 총 checkpoint activation을 얻고, 1.3e11 optimizer state와 함께 거의 정확히 8e12 = 8TB가 됩니다. TPUv5p 슬라이스는 총 `393TB`의 HBM을 가지므로, HBM 제한 아래에 안전하게 있습니다. 다음으로 comm 또는 compute-bound가 될지 살펴보겠습니다. 4096개 칩과 3개의 병렬 axis로, `850 * 4096 = 3.48M` 토큰의 최소 배치 크기를 할 수 있습니다. 이는 3M 배치 크기보다 약간 위입니다. 따라서 실제로 comm-bound이며, 이는 슬픕니다. 따라서 일반적인 답은 **아니오, FSDP만으로는 할 수 없습니다**.

3. 이제 주요 관심사가 comm-bound라는 것을 알므로, 몇 가지 숫자를 넣어보겠습니다. 먼저, 위의 판별식에서 혼합 FSDP + model parallelism을 가진 칩당 배치 크기가 $2 \cdot 2550^2 / F = 940$ 이상이어야 한다는 것을 알고 있으며, 이는 실제로 순수한 FSDP보다 약간 나쁩니다. 분명히 이는 우리가 한 근사치의 일부 때문인데, 이는 혼합 FSDP + model parallelism이 실제로 그리 나아지지 않는다는 것을 시사합니다. 부분적으로 이는 $F$가 너무 작아서 전체 axis 가치의 model parallelism을 할 수 없기 때문입니다. 이를 해결하는 한 가지 방법은 4개 칩의 작은 subring의 tensor parallelism을 하고 첫 번째 axis의 나머지 대역폭을 FSDP에 전용하는 것입니다. 수학을 하지는 않겠지만 아마도 comm-bound 없이 이를 할 수 있다는 것을 확인하는 것이 좋습니다.

{% enddetails %}

**질문 4:** 배치 크기를 1M으로 줄이고 싶다면 어떨까요? 이는 질문 3의 답에 어떤 영향을 미칠까요? 배치 크기 10M은 어떨까요?

<h3 markdown=1 class="next-section">Part 5는 여기까지입니다! 이 내용을 실제 LLaMA 모델에 적용하는 Part 6은 [여기를 클릭하세요](../applied-training-kr)!</h3>

## 부록

### 부록 A - FSDP에 대한 추가 정보

FSDP가 매개변수/gradient를 어떻게 샤딩하는지 보여주는 좋은 추가 그림입니다. 행은 순서대로 순수한 data parallelism, ZeRO-1/2/3입니다. 실질적으로 같은 통신 부하를 가지므로 ZeRO-3를 하지 않을 이유가 별로 없습니다.

{% include figure.liquid path="assets/img/fsdp-figure.png" class="img-fluid" %}

**Figure:** 순수한 data parallelism, ZeRO-1/2/3 각각에 대한 매개변수, gradient, optimizer state 메모리를 보여주는 다이어그램. [Source](https://arxiv.org/abs/1910.02054).

### 부록 B - 역방향 패스에 필요한 comm 도출

위에서, Transformer layer 순방향 패스를 Out\[B, D\] \= In\[B, D\] \*D W<sub>in</sub>\[D, F\] \*<sub>F</sub> W<sub>out</sub>\[F, D\]로 단순화했습니다. 역방향 패스에 필요한 comm을 어떻게 도출할까요?

이는 단일 matmul **Y = X \* A**에 대한 이전 섹션의 규칙에서 상당히 자연스럽게 따라옵니다:

$$\frac{dL}{dA} = \frac{dL}{dY}\frac{dY}{dA} = X^T \left(\frac{dL}{dY}\right)$$

$$\frac{dL}{dX} = \frac{dL}{dY}\frac{dY}{dX} = \left(\frac{dL}{dY}\right) A^T$$

이를 사용하여, 다음 공식을 얻습니다 (Tmp\[B, F\]를 In\[B, D\] \* W<sub>in</sub>\[D, F\]로 나타냄):

<div markdown=1 class="algorithm">

1. dW<sub>out</sub>[F, D] = Tmp[B, F] \*<sub>B</sub> dOut[B, D] 
2. dTmp[B, F] = dOut[B, D] \*<sub>D</sub> W<sub>out</sub>[F, D] 
3. dW<sub>in</sub> = dTmp[B, F] \*<sub>B</sub> Tmp[B, F] 
4. dIn[B, D] = dTmp[B, F] \*<sub>F</sub> W<sub>in</sub>[D, F]

</div>

이들 공식은 샤딩에 대한 언급 없이 수학적 명제입니다. 역방향 패스의 작업은 이 네 가지 양을 계산하는 것입니다. 따라서 필요한 comm을 알아내기 위해, 우리는 위의 네 방정식에서 matmul될 모든 양의 샤딩 (Tmp, dOut, W<sub>out</sub>, W<sub>in</sub>)을 가져오고, 이는 병렬화 방식에 의해 지정되며, 샤딩된 matmul의 규칙을 사용하여 어떤 comm을 해야 하는지 알아냅니다. dOut은 Out과 같은 방식으로 샤딩됩니다.

### 부록 C - 혼합 FSDP + model parallelism에 대한 배치 크기 제약의 대안적 도출

위에서 FSDP + model parallelism의 조합을 사용할 때 다음 조건에서 compute-bound가 될 수 있다고 도출했습니다:

$$ \frac{B}{N} > \frac{4\alpha^2}{M_X M_Y F} $$

다음은 이 사실의 대안적 도출입니다. 통신 시간을 계산 시간과 같게 설정하고, 이 등식을 불가능하게 만드는 조건을 찾는 것으로 시작합니다.

$$\frac{F}{Y \cdot M_X} + \frac{B}{X \cdot M_Y} = \frac{B \cdot F}{N \cdot \alpha}$$

$XY=N$이므로, $X$로 다시 쓸 수 있습니다:

$$\frac{FX}{N \cdot M_X} + \frac{B}{X \cdot M_Y} = \frac{B \cdot F}{N \cdot \alpha}$$, 또는

$$X^2 \frac{F}{N \cdot M_X} + \frac{B}{M_Y} - X \frac{B \cdot F}{N \cdot \alpha} = 0.$$

이는 $X$에서 이차 방정식이므로, 해가 없는 지점은 판별식이 0이 되는 지점입니다. 이는 다음일 때 발생합니다:

$$B^2\cdot F^2 \cdot M_X^2 \cdot M_Y^2 - 4\cdot \alpha^2 \cdot F \cdot B \cdot N \cdot M_Y \cdot M_X = 0$$

또는 단순화하면

$$B\cdot F \cdot M_X \cdot M_Y - 4\cdot \alpha^2 \cdot N = 0$$

이는 다음을 제공합니다:

$$B = \frac{4 \cdot \alpha^2 \cdot N}{F \cdot M_X \cdot M_Y}$$

따라서 총 배치 크기를 총 칩 수로 나눈 것은 다음 아래로 떨어질 수 없습니다:

$$\frac{4 \alpha^2}{F \cdot M_X \cdot M_Y},$$

위에서 도출한 것과 같습니다.