---
layout: distill
title: "알아야 할 모든 Transformer 수학"
# permalink: /main/
description: "여기서는 Transformer 아키텍처를 빠르게 리뷰하고, 특히 FLOP, 바이트, 그리고 기타 관심 있는 수량들을 계산하는 방법을 살펴보겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 4

previous_section_url: "../sharding-kr"
previous_section_name: "Part 3: Sharding"

next_section_url: ../training-kr
next_section_name: "Part 5: Training"

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

bibliography: main.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Counting Dots"
  - subsections:
    - name: "Forward와 reverse FLOP"
  - name: "Transformer Accounting"
  - name: "Global FLOP과 Params 계산"
  - name: "기타 수학"
  - subsections:
    - name: "Sparsity와 Mixture-of-Experts"
    - name: "Gradient checkpointing"
    - name: "Key-Value (KV) caching"
  - name: "이 Section에서 무엇을 얻어가야 하는가?"
  - name: "연습할 몇 가지 문제들"
  - name: "부록"
  - subsections:
    - name: "부록 A: Flash Attention은 어떻게 작동하는가?"

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

## Counting Dots

vector $$x$$,$$y$$와 matrix $$A$$,$$B$$를 다음과 같은 shape으로 시작해보겠습니다:

$$
\def \red#1{\textcolor{red}{#1}}
\def \green#1{\textcolor{green}{#1}}
\def \blue#1{\textcolor{blue}{#1}}
\def \purple#1{\textcolor{purple}{#1}}
\def \orange#1{\textcolor{orange}{#1}}
\def \gray#1{\textcolor{gray}{#1}}

\begin{array}{cc}
\textrm{array}  & \textrm{shape} \\ \hline
x               & \textrm{[P]}   \\
y               & \textrm{[P]}   \\
A               & \textrm{[N P]} \\
B               & \textrm{[P M]} \\
\hline
\end {array}
$$

- $$x \cdot y$$의 dot product는 $$P$$개의 _덧셈_과 _곱셈_을 필요로 하므로, 총 $$2P$$개의 floating-point operation입니다.
- matrix-vector product $$Ax$$는 $$A$$의 행들을 따라 $$N$$개의 dot-product를 수행하므로, $$2NP$$ FLOP입니다.
- matrix-matrix product $$AB$$는 $$B$$의 각 column에 대해 $$M$$개의 matrix-vector product를 수행하므로, 총 $$2NPM$$ FLOP입니다.
- 일반적으로, 일부 dimension은 <span style="color:red">CONTRACTING</span>이고 일부는 <span style="color:blue">BATCHING</span>인 두 고차원 array $$C$$와 $$D$$가 있다면 (예: $$C[\blue{GH}IJ\red{KL}], D[\blue{GH}MN\red{KL}]$$), 이 contraction의 FLOP 비용은 batch와 contraction dimension은 한 번만 계산되는 모든 $$C$$와 $$D$$ dimension의 곱의 2배입니다 (예: $$2\blue{GH}IJMN\red{KL}$$). dimension은 두 multiplicand에 모두 나타날 때만 batching입니다. (contracting dimension이 없고 단순히 elementwise product인 경우 2배 factor는 적용되지 않습니다.)

$$
\begin{array}{ccc}
\textrm{Operation} & \textrm{FLOPs} & \textrm{Data} \\
\hline
x \cdot y  & 2P   & 2P      \\
A x        & 2NP  & NP + P  \\
AB         & 2NPM & NP + PM \\
[c_0,...,c_N] \cdot [d_0,...,d_N] &
2 \prod c_i \times \prod_{\substack{d_j \notin \blue{BATCH} \\ d_j \notin \red{CONTRACT}}} d_j
&
  \prod c_i + \prod d_j \\
\hline
\end {array}
$$

matrix-matrix multiply의 경우 *compute*는 3차 $$O(N^3)$$로 scale되지만 data transfer는 2차 $$O(N^2)$$로만 scale된다는 사실을 주목하세요 \- 이는 matmul 크기를 키울수록 compute-saturated limit에 도달하기가 *더 쉬워진다*는 의미입니다. 이는 매우 특이한 현상이며, 우리가 matrix multiplication이 지배적인 architecture를 사용하는 이유를 크게 설명합니다 \- 이들은 scaling에 적합하기 때문입니다!

{% include figure.liquid path="assets/img/matmul-flops.gif" class="img-fluid" %}

### Forward와 reverse FLOP

training 중에는 주어진 matrix multiply의 결과를 특별히 신경 쓰지 않습니다; 우리가 정말 관심 있는 것은 그 derivative입니다. 이는 backpropagation 중에 훨씬 더 많은 FLOP을 수행한다는 의미입니다.

**B**가 더 큰 network의 한 matrix이고 **A**가 우리의 input activation이며 **C = A B**라고 상상해보면, **B**에 대한 loss **L**의 derivative는 chain rule에 의해 주어집니다:

$$\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial B} = A^T \left(\frac{\partial L}{\partial C}\right)$$

이는 outer product이며 계산하는 데 $2NPM$ FLOP이 필요합니다 ($N$ dimension에 걸쳐 contract하므로). 마찬가지로, **A**에 대한 loss의 derivative는

$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C}\frac{\partial C}{\partial A} = \left(\frac{\partial L}{\partial C}\right) B^T$$

**dL/dC**가 크기 $$[N, M]$$의 (co-)vector이므로 다시 $2NPM$ FLOP입니다. 이 quantity는 parameter에 대한 derivative는 아니지만, network의 이전 layer들에 대한 derivative를 계산하는 데 사용됩니다 (예: dL/dC가 위에서 dL/dB를 계산하는 데 사용되는 것처럼).

이들을 합하면, **training 중에는 총 6NPM FLOP을 가진다**는 것을 알 수 있습니다. inference 중 2NPM과 비교하면: forward pass에서 2NPM, backward pass에서 4NPM입니다. PM이 matrix의 parameter 수이므로, 이는 training 중 Transformer FLOP의 유명한 $$6 * \text{num parameters} * \text{num tokens}$$ 근사의 가장 간단한 형태입니다: 각 token은 $$6 * \text{num parameters}$$ FLOP을 필요로 합니다. 아래에서 더 정확한 도출을 보여드리겠습니다.

## Transformer Accounting

Transformer는 미래입니다. 음, 적어도 현재입니다. 아마 몇 년 전에는 여러 architecture 중 하나였을 것입니다. 하지만 오늘날에는 architecture의 거의 모든 세부사항을 아는 것이 가치가 있습니다. 우리는 architecture를 다시 소개하지 않겠지만 [이 블로그](https://jalammar.github.io/illustrated-transformer/)와 [원본 Transformer 논문](https://arxiv.org/abs/1706.03762)이 도움이 될 수 있는 참고자료입니다.

다음은 Transformer decoder architecture의 기본 다이어그램입니다:

{% include figure.liquid path="assets/img/transformer-diagram.png" class="img-fluid" caption="<b>Figure:</b> 이 다이어그램은 표준 Transformer의 한 layer를 보여주며 위에서 아래로 흐릅니다. 우리는 Transformer에서 array의 shape과 layout을 설명하기 위해 single-letter 규약을 사용하며, contracting dimension을 빨간색으로, batched dimension을 파란색으로 다시 보여줍니다. 주어진 operation에서 input shape은 왼쪽 위에, parameter shape은 오른쪽 위에 주어지며, 결과 shape은 아래에 있습니다. 예를 들어 BTD는 gating einsum의 input shape이고 DF는 weight shape입니다." %}

**참고 [gating einsum]**: 위 다이어그램은 up-projection matrix를 두 matrix ($W_\text{In1}$과 위의 $W_\text{In2}$)로 분할하여 그 output을 일종의 "gating function"으로 elementwise 곱하는 "[gating einsums](https://arxiv.org/abs/2002.05202)"<d-cite key="glu"></d-cite>를 사용합니다. 모든 LLM이 이를 사용하는 것은 아니므로, 때때로 단일 $W_\text{In}$ matrix와 총 MLP parameter count가 3DF 대신 2DF인 경우를 볼 수 있습니다. 일반적으로 이 경우 D와 F는 parameter count를 3 matrix case와 동일하게 유지하도록 scale up됩니다. 그렇긴 하지만, 어떤 형태의 gating einsum은 LLAMA, DeepSeek, 그리고 많은 다른 model들에서 사용됩니다.

**참고 2 [MHA attention]**: self-attention에서는 T와 S가 동일하지만 cross-attention에서는 다를 수 있습니다. vanilla Multi-Head Attention (MHA)에서는 N과 K가 동일하고, [Multi-Query Attention](https://arxiv.org/abs/1911.02150) (MQA)<d-cite key="mqa"></d-cite>에서는 K=1이며, [Grouped MQA](https://arxiv.org/abs/2305.13245) (GMQA)<d-cite key="gmqa"></d-cite>에서는 K가 단순히 N을 나누어야 합니다.

## Global FLOP과 Params 계산

아래에서는 **L** factor를 모든 곳에 붙이는 것을 피하기 위해 per-layer FLOP을 계산하겠습니다.

### MLP

Transformer의 MLP는 일반적으로 element-wise로 결합되는 2개의 input matmul과 단일 output matmul로 구성됩니다:

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{in1}[\red{D}, F] & 6BTDF & DF \\[10pt]
A[B,T,\red{D}] \cdot W_{in2}[\red{D}, F] & 6BTDF & DF \\[10pt]
\sigma\left(A_{in1}\right)[B,T, F] * A_{in2}[B,T, F] & \gray{O(BTF)} \\[10pt]
A[B,T,\red{F}] \cdot W_{out}[\red{F}, D] & 6BTDF & DF \\[10pt]
\hline \\
& \approx 18BTDF & 3DF
\end{array}
$$

### Attention

서로 다른 **Q**와 **KV** head 수를 가진 일반적인 grouped-query attention case에 대해, **Q**,**K**,**V** projection에 대해 동일한 head dimension H를 가정하고, **QKVO** matmul의 비용을 추정해봅시다:

$$
\begin{array}{ccc}
\textrm{operation} & \textrm{train FLOPs} & \textrm{params} \\
\hline \\
A[B,T,\red{D}] \cdot W_{Q}[\red{D}, N, H] & 6BTDNH & DNH \\[10pt]
A[B,T,\red{D}] \cdot W_{K}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{D}] \cdot W_{V}[\red{D}, K, H] & 6BTDKH & DKH \\[10pt]
A[B,T,\red{N}, \red{H}] \cdot W_{O}[\red{N}, \red{H}, D] & 6BTDNH & DNH \\[10pt]
\hline \\ & 12BTD(N+K)H & 2D(N+K)H
\end{array}
$$

dot-product attention operation은 더 미묘하며, 효과적으로 $$B$$, $$K$$ dimension에 걸쳐 batch된 $$TH \cdot HS$$ matmul, softmax, 그리고 다시 $$B$$, $$K$$ dimension에 걸쳐 batch된 $$TS \cdot SH$$ matmul입니다. batched dim을 파란색으로 강조합니다:

$$
\begin{array}{cc}
\textrm{operation} & \textrm{train FLOPs} \\
\hline \\[3pt]
Q[\blue{B}, T, \blue{K}, G, \red{H}] \cdot K[\blue{B}, S, \blue{K}, \red{H}]
& 6BTSKGH = 6BTSNH  \\[3pt]
\textrm{softmax}_S \;\; L[B, T, S, K, G] & \gray{O(BTSKG) = O(BTSN)} \\[3pt]
S[\blue{B}, T, \red{S}, \blue{K}, G] \cdot V[\blue{B}, \red{S}, \blue{K}, H] 
& 6BTSKGH = 6BTSNH \\[3pt]
\hline \\
& \approx 12BTSNH = 12BT^2NH \\
\end{array}
$$

### 기타 Operation

Transformer에서 일어나는 여러 다른 operation이 있습니다. Layernorm은 비교적 저렴하고 1차 비용 추정에서는 무시될 수 있습니다. 또한 최종적인 거대한 (layer별이 아닌) unembedding matrix multiply가 있습니다.

$$
\begin{array}{ccc}
\textsf{operation} & \textsf{train FLOPs} & \textsf{params} \\
\hline \\
\textrm{layernorm}_D \;\; A[B,T,\red{D}] & \gray{O\left(BTD\right)} & \gray{D} \\[10pt]
A[B,T,\red{D}] \cdot W_{unembed}[\red{D}, V] & 6BTDV & DV \\
\end{array}
$$

### Transformer FLOP에 대한 일반적인 경험 법칙

더 짧은 context training에 대해 dot-product attention의 비용을 무시한다면, 모든 layer에 걸친 총 FLOP은

$$
\begin{align*}
(18BTDF + 12BTD(N+K)H)L = 6 *BT * (3DF + 2D(N+K)H)L \\ = 6 * \textrm{num tokens} * \textrm{parameter count}
\end{align*}
$$

attention FLOP을 무시한 dense Transformer FLOP count를 추정하는 유명한 경험 법칙으로 이어집니다. (Unembedding은 $6BSDV$ FLOP과 $DV$ param을 가진 또 다른 간단한 matmul이며, 동일한 경험 법칙을 따릅니다.)

### context length에 따른 attention의 fractional cost

위에서 dot-product attention을 고려하고 $$F=4D$$, $$D=NH$$ (일반적인 경우), 그리고 $$N=K$$라고 가정하면:

$$\small{\frac{\textrm{attention FLOPs}}{\textrm{matmul FLOPs}} = \frac{12BT^2NH}{18BTDF + 24BTDNH} = \frac{12BT^2D}{4*18 BTD^2 + 24 BTD^2} = \frac{12BT^2D}{96 BTD^2} = \frac{T}{8D}}$$

따라서 핵심은 **dot-product attention FLOP은 T>8D일 때만 training 중에 dominant하게 된다**는 것입니다. D ~ 8k의 경우, 이는 ~64K token입니다. 이는 MLP 크기가 증가할수록 attention FLOP이 덜 중요해진다는 의미이므로 어느 정도 이해가 됩니다. 큰 model의 경우, attention의 2차 비용은 실제로 더 긴 context training에 큰 장애물이 아닙니다. 하지만 더 작은 model의 경우, 예를 들어 Gemma-27B에서 D=4608이므로 attention은 32k sequence length 근처에서 dominant하게 됩니다. Flash Attention도 [부록 A](#부록-a-flash-attention은-어떻게-작동하는가)에서 간략히 논의하는 long-context의 비용을 완화하는 데 도움이 됩니다.

## 기타 수학

### Sparsity와 Mixture-of-Experts

표준 Transformer의 단일 dense MLP block을 동적으로 routing할 수 있는 독립적인 MLP set으로 대체하는 Mixture of Experts (MoE) model<d-cite key="moe"></d-cite>에 대해 간략히 논의하지 않을 수 없습니다. 첫 번째 근사에서, **MoE는 단지 하나 대신 layer당 E개의 MLP block을 가진 일반적인 dense model**입니다. 각 token은 이러한 expert 중 $k$개를 활성화하며, 일반적으로 $k=2$입니다. 이는 parameter count를 $O(E)$만큼 증가시키면서, dense version과 비교하여 token당 총 활성화된 parameter 수를 $k$배 곱합니다.

{% include figure.liquid path="assets/img/moe.png" class="img-fluid img-small" caption="<b>Figure:</b> $n$개 expert를 가진 MoE layer의 예시입니다. gating expert는 각 token을 그 중 $k$개로 route하고, 해당 $k$개 MLP의 output이 합해집니다. 우리의 parameter count는 각 expert 크기의 $n$배이지만, 각 token에 대해 $k$개만 사용됩니다. <a href=\"https://deepgram.com/learn/mixture-of-experts-ml-model-guide\">Source</a>." %}

dense model과 비교하여, MoE는 주로 token을 올바른 expert로 route하고 다시 home device로 가져오는 두 개의 AllToAll (MoE block 전후 각각 하나)인 새로운 comm을 도입합니다.<d-footnote>기술적으로, 이는 우리가 expert와 동일한 axis를 따라 data 또는 sequence sharded된 경우에만 발생합니다.</d-footnote> 하지만 이전 section에서 본 바와 같이, 각 AllToAll의 비용은 single axis를 따른 comparable AllGather의 1/4에 불과합니다 (bidirectional ring의 경우).

### Gradient checkpointing

algorithm으로서의 Backpropagation은 memory를 compute와 교환합니다. backward pass가 $$O(n_\text{layers}^2)$$ FLOP을 필요로 하는 대신, **$$O(n_\text{layers})$$ memory를 필요로 하여**, forward pass 중에 생성된 모든 중간 activation을 저장합니다. 이는 2차 compute보다 낫지만, memory-wise로는 매우 비쌉니다: $$B * T=4M$$ (batch당 총 4M token), L=64, 그리고 D=8192인 model이 모든 불필요한 backward pass compute를 피한다면 bfloat16에서 대략 $$2 * 20 * B * T * D * L = 84TB$$의 activation을 저장해야 합니다. 20은 위 Transformer 다이어그램의 모든 중간 node를 (대략) 계산한 것에서 나오며, 예를 들어

$$f(x) = \exp(g(x))$$

$$\frac{df}{dx} = \exp(g(x)) \cdot \frac{dg}{dx}$$

따라서 재계산을 피하려면 forward pass에서 $$g(x)$$와 $$\exp(g(x))$$를 저장해야 합니다. 이렇게 많은 memory 저장을 피하기 위해, 중간 activation의 일부분만 저장하도록 선택할 수 있습니다. 우리가 사용하는 몇 가지 strategy가 있습니다.

* **Block remat**: 각 layer의 input만 저장합니다. 이는 우리가 사용하는 가장 공격적인 방법이며 layer당 1개의 checkpoint만 저장하므로, 위 예시에서 4.2TB만 저장하게 됩니다. 이는 backward pass에서 본질적으로 모든 forward pass FLOP을 반복하도록 강제하므로, FLOP을 $$6ND$$에서 대략 $$8ND$$로 증가시킵니다.
* **Big matmul만:** 또 다른 간단한 정책은 큰 matmul의 output만 저장하는 것입니다. 이는 backward pass 중에 큰 matmul을 재계산하는 것을 피할 수 있게 하지만, 여전히 다른 activation function과 attention의 일부를 재계산하게 만듭니다. 이는 layer당 20을 7에 더 가깝게 줄입니다.

이는 결코 포괄적이지 않습니다. JAX를 사용할 때, 이들은 일반적으로 `jax.remat`/`jax.checkpoint`에 의해 제어됩니다 ([여기](https://jax.readthedocs.io/en/latest/_autosummary/jax.checkpoint.html)에서 더 읽을 수 있습니다).

### Key-Value (KV) caching

[Section 7](../inference-kr)에서 보게 되겠지만, LLM inference는 prefill과 generation이라는 두 가지 주요 부분을 가집니다.

* **Prefill**은 긴 prompt를 처리하고 generation에서 사용하기 위해 attention activation을 Key-Value Cache (KV Cache)에 저장합니다. 특히 attention block의 key-value projection입니다.
* **Generation**은 이러한 KV cache 여러 개를 함께 batch하고 각각에서 token을 sampling합니다.

각 KV cache는 효과적으로 크기 $[2, S, L, K, H]$의 array이며, 여기서 2는 key와 value를 고려한 것입니다. 이는 꽤 큽니다! Key-Value cache의 총 크기는 int8에서 $2SLKH$입니다. 8k context length, 64 layer, 그리고 $KH = NH = D = 8192$인 중간 크기 model의 경우, 이는 $2 \cdot 8192 \cdot 64 \cdot 8192 = 8\text{GiB}$입니다. $K \ll N$인 GMQA를 원하는 이유를 알 수 있습니다.

## 이 Section에서 무엇을 얻어가야 하는가?

* Transformer의 전반적인 parameter와 FLOP은 계산하기 상당히 쉬우며, MHA를 가정하여 여기에 요약되어 있습니다 (batch size B, vocab size V, length T의 sequence, D=d<sub>model</sub>, 그리고 F=d<sub>ff</sub>):


<!-- $$
\begin{array}{ccc}
\textrm{Component} & \textrm{Params per layer} & \textrm{Training FLOPs per layer} \\
\hline \\
\textbf{MLP} & 3DF & 18BTDF \\[10pt]
\textbf{Attention} & 4DNH & 24BTDNH + 12BT^2NH \\[10pt]
\textbf{Other} & D & BTD \\[10pt]
\textbf{Vocab} & DB \text{ (total, not per-layer)} & 12BTDV \\[10pt]
\end{array}
$$ -->


| Component     | Params per layer          | Training FLOPs per layer      |
| :------------ | :------------------------ | :---------------------------- |
| **MLP**       | 3DF                       | 18BTDF                        |
| **Attention** | 4DNH                      | 24BTDNH \+ 12BT<sup>2</sup>NH |
| **Other**     | D                         | BTD                           |
| **Vocab**     | DV (total, not per-layer) | 12BTDV                        |

* MLP block의 parameter count가 총 parameter count를 지배하고, sequence length $T < 8D$인 한 MLP block이 FLOP budget도 지배합니다.
* training 중 총 FLOP budget은 합리적인 context length에 대해 $$6 \cdot \text{num_params} \cdot \text{num_tokens}$$로 잘 근사됩니다.
* inference 중에, 우리의 KV cache는 cache당 대략 $$2 \cdot S \cdot L \cdot N \cdot H$$이지만, architectural modification으로 종종 이를 줄일 수 있습니다.

## 연습할 몇 가지 문제들

**Question 1:** $D=4096$, $F=4 \cdot D$, $V=32,000$, 그리고 $L=64$인 model은 얼마나 많은 parameter를 가지나요? 이 중 attention parameter의 비율은 얼마나 되나요? token당 KV cache는 얼마나 큰가요? *$N\cdot H=D$이고 int8 KV를 가진 multi-head attention이라고 가정할 수 있습니다.*

{% details 답을 보려면 여기를 클릭하세요. %}

1. 총 parameter는 대략 $$L \cdot (3DF + 4DNH + D) + 2DV$$입니다. 주어진 수치들에 대해, 이는 $$64 \cdot (3 \cdot 4e3 \cdot 16e3 + 4 \cdot 4e3 \cdot 4e3 + 4e3) + 2 \cdot 4e3 \cdot 32e3 = 16e9$$, 즉 16B parameter입니다.
2. 일반적으로 attention parameter 대 총 parameter의 비율은 $$4DNH / (4DNH + 3DF) = 4D^2 / (4D^2 + 12D^2) = 1/4$$입니다. 이는 대략 1/4의 parameter가 attention에 사용된다는 것을 의미합니다.
3. token당, 우리의 KV cache는 int8에서 $$2 \cdot L \cdot N \cdot H = 2 \cdot 64 \cdot 4096$$이며, 이는 `512kB / token`입니다.

{% enddetails %}

**Question 2:** `{'X': 4, 'Y': 8, 'Z': 4}`에서 A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F]를 수행하는 데 필요한 총 FLOP은 얼마나 되나요? 각 TPU는 얼마나 많은 FLOP을 수행하나요?

{% details 답을 보려면 여기를 클릭하세요. %}

operation의 총 "이론적" FLOP은 $$2 \cdot B \cdot D \cdot F$$입니다. 하지만 computation이 Z dimension에 걸쳐 sharded되지 않기 때문에, 우리는 실제로 Z만큼의 추가 FLOP을 수행하고 있으므로, 총 $$2 \cdot B \cdot D \cdot F \cdot Z$$ FLOP입니다. computation이 다른 dimension들에 걸쳐 sharded되므로, device당 총은 대략 $$2 \cdot B \cdot D \cdot F / (X \cdot  Y)$$입니다.

{% enddetails %}

**Question 3:** $A[I,J,K,L] * B[I,J,M,N,O] \rightarrow C[K,L,M,N,O]$를 수행하는 데 몇 개의 FLOP이 관련되나요?

{% details 답을 보려면 여기를 클릭하세요. %}

위 규칙에 따르면, I와 J가 contracting dimension이고 K, L, M, N, O가 non-contracting dimension입니다. "batching dimension"은 없으므로, 이는 단순히 모든 axis의 합인 $$2 \cdot I \cdot J \cdot K \cdot L \cdot M \cdot N \cdot O$$입니다. 공유 axis가 있다면, 한 번만 계산됩니다.

{% enddetails %}

**Question 4:** self-attention (Q/K/V/O projection 무시)의 arithmetic intensity는 무엇인가요? *Q와 KV length T와 S의 function으로 답을 주세요.* 어떤 context length에서 attention이 FLOP-bound인가요? TPU의 HBM bandwidth를 고려하여, context length가 증가함에 따라 FFW block에 대한 attention의 효과적인 상대 비용을 plot하세요.

{% details 답을 보려면 여기를 클릭하세요. %}

Self-attention은 $$Q$$, $$K$$, 그리고 $$V$$ activation을 load한 다음, $$\text{softmax}(Q \cdot K) \cdot V$$를 계산하고, 결과를 다시 HBM에 write해야 합니다. 이는 Flash Attention으로 수행되므로 이 수학에는 몇 가지 주의사항이 있지만, 기본적으로 bf16에서 self-attention은 

$$\text{Q[B,T,N,H]} \rightarrow_\text{reshape} \text{Q[B, T, K, G, H]} \cdot \text{K[B, S, K, H]} \rightarrow \text{O[B, T, S, K, G]}$$

$$U=\text{softmax}_S(\text{O[B, T, S, K, G]})$$

$$\text{U[B, T, S, K, G]} \cdot \text{V[B, S, K, H]} \rightarrow \text{X[B, T, K, G, H]}$$

따라서 총 바이트는 $$2 * \text{sizeof}(Q) + 2 * \text{sizeof(K or V)} = 4BTNH + 4BSKH = 4BHK * (TG + S)$$이고, 총 FLOP은 $$4BTSNH + O(BTSN)$$이며 arithmetic intensity는 $$4BTSKGH / (4BHK * (TG + S))$$입니다.

따라서 기본적으로, prefill 중에는 $$S=T$$이므로 arithmetic intensity가 $$4BT^2KGH / 4BHKT \cdot (G+1) = TG/(G + 1) = O(T)$$입니다. generation 중에는 $$T=1$$이므로 $$4BSKGH / (4BHK \cdot (G + S)) = SG / (G + S) \rightarrow G$$ ($$S$$가 매우 크다고 가정)입니다. 질문을 어떻게 해석하느냐에 따라, prefill 또는 training 중 self-attention은 sequence sharding이 없다고 가정할 때 S=240에서 compute bound입니다. generation 중에는 $$G$$가 작기 때문에 결코 compute bound가 아닙니다. 그럼에도 불구하고, $$G$$를 증가시키면 compute bound에 더 가까워진다는 것을 볼 수 있습니다.

{% enddetails %}

**Question 5:** self-attention FLOP이 QKVO projection FLOP과 같아지는 sequence length는 얼마인가요?

{% details 답을 보려면 여기를 클릭하세요. %}

이는 순전히 $$24BTDNH == 12BT^2NH$$인 때의 문제입니다. 단순화하면 $$2D = T$$를 얻으므로, 예를 들어 $$D=4096$$의 경우, 이는 $$8192$$입니다. 이는 대부분의 합리적인 context length에 대해 matmul FLOP이 더 크다는 것을 알려줍니다.

{% enddetails %}

**Question 6:** forward pass 중에 Transformer layer의 7개 주요 matmul (Q, K, V, O \+ 3개 FFW matrix)의 output만 저장한다고 하세요. backward pass 중 "rematerialize"하기 위해 얼마나 많은 추가 FLOP이 필요한가요?

**Question 7:** DeepSeek v3는 14.8T token에 대해 2.79M H800 시간 동안 훈련되었다고 합니다 ([source](https://arxiv.org/pdf/2412.19437v1)). 37B activated parameter를 가지고 있다는 점을 고려할 때, 대략적으로 어떤 hardware utilization을 달성했나요? *힌트: 그들이 structured sparsity 없이 FP8 FLOP을 사용했다는 점에 주목하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

[여기](https://lenovopress.lenovo.com/lp1814.pdf) spec sheet에서, sparsity를 가진 FP8 성능은 3,026 TFLOPs/s이거나, sparsity 없이는 일반적으로 이의 절반인 (`1.513e15` FLOPs/s)임을 찾을 수 있습니다. 2.79M H800 시간은 `2.79e6 * 1.513e15 * 60 * 60 = 1.52e25` 총 FLOP을 의미합니다. 37B의 activated parameter count를 고려하면, 이 training run은 약 `6 * 37e9 * 14.8e12 = 3.3e24` FLOP을 사용했어야 합니다. 이는 FLOP utilization이 약 `3.3e24 / 1.52e25 = 21.7%`라는 것을 의미합니다.

{% enddetails %}

**Question 8:** Mixture of Experts (MoE) model은 표준 dense MLP block의 $E$개 사본을 가지며, 각 token은 이러한 expert 중 $k$개를 활성화합니다. TPU v5e에서 int8 weight를 가진 MoE가 compute-bound가 되기 위해 필요한 token 단위의 batch size는 얼마인가요? 256개 (routed) expert와 $k=8$을 가진 DeepSeek의 경우, 이 숫자는 얼마인가요?

{% details 답을 보려면 여기를 클릭하세요. %}

각 expert의 $E$개 사본을 가지므로, int8에서 $E \cdot D \cdot F$ 바이트를 load해야 합니다. 각 token이 $k$개 expert를 활성화하므로, $2\cdot k \cdot B \cdot D \cdot F$ FLOP을 가집니다. bfloat16 FLOP으로 compute-bound가 되려면, $(2\cdot k \cdot BDF) / EDF > 240$ 또는 $k \cdot B / E > 120$일 때 발생하는 240 이상의 arithmetic intensity가 필요합니다.

따라서, compute bound가 되려면 $B > 120 \cdot E / k$가 필요합니다. DeepSeek의 경우, 이는 $B > 120 \cdot 256 / 8 = 3840$를 제공합니다. 이는 generation 시에 놀랍도록 큰 batch size입니다.

{% enddetails %}

<h3 markdown=1 class="next-section">Part 4는 여기까지입니다! Part 5 (Transformer training scaling에 대한)는 [여기](../training-kr)를 클릭하세요!</h3>

## 부록

### 부록 A: Flash Attention은 어떻게 작동하는가?

Transformer를 매우 긴 context로 scaling하는 것에 대한 전통적인 반대는 attention FLOP과 memory 사용량이 context length에 따라 2차적으로 scale된다는 것입니다. attention QK product가 B는 batch size, S와 T는 Q와 K sequence dim, N은 head 수인 $[B, S, T, N]$ shape을 가진다는 것은 사실이지만, 이 주장에는 몇 가지 심각한 주의사항이 있습니다:

1. Section 4에서 주목했듯이, 이것이 2차적이긴 하지만, attention FLOP은 $$S > 8 \cdot D$$일 때만 dominant하며, 특히 training 중에 단일 attention matrix의 memory는 sharded될 때 특히 memory에 있는 모든 weight와 activation checkpoint에 비해 작습니다.
2. attention을 계산하기 위해 전체 attention matrix를 materialize할 필요가 없습니다! local sum과 max를 계산하고 array의 작은 chunk보다 더 많이 materialize하는 것을 피할 수 있습니다. 총 FLOP은 여전히 2차적이지만, memory pressure를 drastically 줄입니다.

이 두 번째 관찰은 [Rabe et al. 2021](https://arxiv.org/abs/2112.05682)에 의해 처음 이루어졌고 나중에 [Flash Attention paper](https://arxiv.org/abs/2205.14135) (Dao et al. 2022)에서 나왔습니다. 기본 아이디어는 K/V의 chunk로 attention을 계산하는 것인데, local softmax와 일부 auxiliary statistic을 계산한 다음, 이를 local chunk와 결합하는 다음 chunk로 전달합니다. 구체적으로, 우리는 다음을 계산합니다:

1. **M:** sequence dimension에 걸친 $$q \cdot k$$의 running max
2. **O:** sequence dimension에 걸친 running full attention softmax
3. **L:** running denominator $$\sum_i (q \cdot k_i - \text{running max})$$

이들을 통해, 우리는 일정한 양의 memory만으로 새로운 max, 새로운 running sum, 그리고 새로운 output을 계산할 수 있습니다. 이것이 어떻게 작동하는지에 대한 대략적인 설명을 제공하자면, attention은 대략 이 operation입니다:

$$\text{Attn}(Q, K, V) = \sum_i \frac{\exp(Q \cdot K_i - \max_j Q \cdot K_j) V_i}{\sum_l \exp(Q \cdot K_l - \max_j Q \cdot K_j)}$$

max는 numerical stability를 위해 빼지며 $$\sum_i \exp(a_i + b) = \exp(b) \sum \exp(a)$$이므로 결과에 영향을 주지 않고 더할 수 있습니다. 위의 분모만 보면, key vector의 두 연속적인 chunk, $$K^1$$과 $$K^2$$를 가지고 있다고 상상하고 각각에 대해 local softmax sum $$L^1$$과 $$L^2$$를 계산합니다:

$$L^1 = \sum_i \exp(Q \cdot K_i^1 - \max_j Q \cdot K_j^1)$$

$$L^2 = \sum_i \exp(Q \cdot K_i^2 - \max_j Q \cdot K_j^1)$$

그러면 다음을 사용하여 이 두 chunk를 함께 full softmax sum으로 결합할 수 있습니다:

$$L^\text{combined} = \exp(M^1 - \max(M^1, M^2)) \cdot L^1 + \exp(M^2 - \max(M^1, M^2)) \cdot L^2$$

여기서

$$M^1 = \max_j Q \cdot K_j^1 \text{ and } M^2 = \max_j Q \cdot K_j^2$$

이는 full softmax에 대해서도 수행될 수 있어, 임의로 큰 softmax sum을 accumulate하는 방법을 제공합니다. 다음은 Flash Attention paper의 전체 algorithm입니다.

{% include figure.liquid path="assets/img/flash-algo.png" class="img-fluid" %}

hardware 관점에서, 이는 Q의 chunk를 VMEM (위 algorithm이 on-chip SRAM이라고 부르는 것)에 맞출 수 있게 하여 각 iteration에서 KV chunk만 load하면 되므로, arithmetic intensity를 줄입니다. 또한 running statistic을 VMEM에 유지할 수 있습니다.

강조할 가치가 있는 마지막 미묘한 점은 training을 위한 Flash VJP (reverse mode derivative) 계산을 실용적으로 만드는 데 사용되는 attention softmax property입니다. 중간 softmax array를 다음과 같이 정의하면:

$$S_{ij} = \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_j}}$$

attention에서, 우리는 reverse-mode *dO*와 *V* array에서 *dS*를 얻습니다:

$$dS_{ij} = dO_{id} \cdot_d V_{jd} = \sum_d dO_{id} V_{jd}$$

이 gradient를 Q와 K로 backpropagate하는 동안

$$d(q_i \cdot k_j) = (dS_{ij} - S_{ij} \cdot_j dS_{ij}) S_{ij}$$

우리는 큰 key **length** dimension을 따른 contraction을 feature **depth** dimension을 따른 local contraction과 교환할 수 있게 하는 identity를 exploit합니다.

$$\begin{align*}
S_{ij} \cdot_j dS_{ij} &= \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} \sum_d dO_{id} V_{jd} \\
&= \sum_d dO_{id} \sum_j \frac{e^{\tau q_i \cdot k_j}}{\sum_k e^{\tau q_i \cdot k_k}} V_{jd} \\
&= \sum_d dO_{id} O_{id} \\
&= dO_{id} \cdot_d O_{id}
\end{align*}$$

이 교체는 VJP에 대한 sequence-block *local* 계산을 구현할 수 있게 하는 데 중요하며, ring attention과 같은 더 영리한 sharding scheme을 가능하게 합니다.