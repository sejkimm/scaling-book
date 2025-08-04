---
layout: distill
title: "Transformer Inference의 모든 것"
# permalink: /main/
description: "Transformer에서 inference를 수행하는 것은 training과 매우 다를 수 있습니다. 부분적으로는 inference가 고려해야 할 새로운 요소인 latency를 추가하기 때문입니다. 이 섹션에서는 모델에서 단일 새 token을 sampling하는 것부터 inference engine의 일부로 여러 accelerator slice에 걸쳐 대형 Transformer를 효율적으로 확장하는 것까지 모든 과정을 다룰 것입니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 7

previous_section_url: "../applied-training-kr"
previous_section_name: "Part 6: LLaMA Training"

next_section_url: ../applied-inference-kr
next_section_name: "Part 8: LLaMA Serving"

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
  - name: "Transformer Inference의 기본"
  - subsections:
    - name: "실제로 우리가 최적화하고자 하는 것은?"
    - name: "Linear operation: 무엇이 bottleneck인가?"
    - name: "Attention은 어떤가?"
    - name: "LLM latency와 throughput의 이론적 추정"
    - name: "Memory는 어떤가?"
    - name: "LLaMA 2-13B의 throughput과 latency 모델링"
  - name: "Generation Throughput과 Latency 개선 기법"
  - name: "여러 Accelerator에 걸친 Inference 분산"
  - subsections:
    - name: "Prefill"
    - name: "Generation"
    - name: "KV cache sharding"
  - name: "효과적인 Inference Engine 설계"
  - subsections:
    - name: "Continuous Batching"
    - name: "Prefix Caching"
    - name: "구현 예시: JetStream"
  - name: "실습 문제"
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

## Transformer Inference의 기본

Transformer를 training하고 이를 사용해 새로운 sequence를 생성하고 싶다고 하겠습니다. _결국 벤치마크 점수 상승과 loss curve 하락은 실제로 무언가 흥미로운 일이 일어날지에 대한 대리 지표일 뿐입니다!_<d-footnote>역사적으로 Transformer에 대한 많은 연구를 적절한 KV cache나 generation loop 구현 없이 inference를 건드리지 않고도 수행할 수 있었습니다. LLM loss나 multiple choice benchmark는 효율적으로 실행할 수 있었기 때문입니다. 이는 특히 연구 코드베이스에서 inference codepath에 많은 개선의 여지가 있다는 것을 의미했습니다.</d-footnote>

Sampling은 개념적으로 간단합니다. sequence를 입력하면 우리가 선호하는 Transformer가 $$\log p(\text{next token}_i \vert \text{previous tokens})$$, 즉 모든 가능한 다음 token에 대한 log-probability를 출력합니다. 이 분포에서 sampling하여 새로운 token을 얻을 수 있습니다. 이 token을 추가하고 이 과정을 반복하면 prompt의 continuation인 token sequence를 얻습니다.

{% include figure.liquid path="assets/img/naive-inference.png" class="img-fluid" caption="<b>그림:</b> Transformer에서의 naive sampling. 파란색 logit이 다음 token에 대한 분포를 제공하며 이로부터 sampling할 수 있습니다. 각 단계마다 전체 prefix를 재처리하므로 algorithm이 $\Theta(n^2)$ runtime을 갖는다는 점에 주목하십시오." %}

우리는 방금 Transformer sampling의 naive 구현을 설명했는데, 작동하기는 하지만 **실제로는 절대 사용하지 않습니다**. token을 생성할 때마다 전체 sequence를 재처리하기 때문입니다. 이 algorithm은 $$n$$개의 token을 생성하는 데 FFW에서 $$O(n^2)$$, attention mechanism에서 $$O(n^3)$$입니다!

**이를 어떻게 피할 수 있을까요?** 매번 전체 forward pass를 수행하는 대신, 각 forward pass에서 일부 중간 activation을 저장하여 이전 token을 재처리하는 것을 피할 수 있습니다. 구체적으로, 주어진 token이 dot-product attention 중에 이전 token에만 attend하므로, 각 token의 key와 value projection을 **KV cache**라는 새로운 data structure에 쓸 수 있습니다. 과거 token에 대한 이러한 key/value projection을 저장하면, 미래 token들은 이전 token에 대한 새로운 FLOP을 수행하지 않고도 $$q_i \cdot k_j$$ product를 계산할 수 있습니다. 훌륭합니다!

이를 염두에 두고, inference는 두 가지 핵심 부분으로 구성됩니다:

* <b style="color: red;">Prefill</b>: 긴 prompt가 주어지면, prompt의 모든 token을 동시에 처리하고 결과 activation(구체적으로는 key-value projection)을 **"KV cache"**에 저장합니다. 또한 마지막 token의 logit을 저장합니다.
* <b style="color: blue;">Generation</b>: KV cache와 이전 logit이 주어지면, logit에서 점진적으로 하나의 token을 sampling하고, 그 token을 다시 Transformer에 입력하여 다음 단계를 위한 새로운 logit set을 생성합니다. 또한 그 새로운 token의 KV activation을 KV cache에 추가합니다. 특별한 `<EOS>` token에 도달하거나 최대 길이 제한에 도달할 때까지 이를 반복합니다.

다음은 KV cache를 사용한 sampling의 diagram입니다:

{% include figure.liquid path="assets/img/cached-inference.png" class="img-fluid" caption="<b>그림:</b> KV cache를 사용한 효율적인 Transformer sampling의 diagram. <b style=\"color: red;\">Prefill</b>은 prompt를 처리하고 모든 per-token key-value activation을 cache에 저장합니다. <b style=\"color: blue;\">Generation</b>은 이 cache(와 마지막 token logit)를 가져와서 새로운 token을 sampling하고, 그 새 token을 모델에 통과시켜 KV cache에 attend하고 새 token의 key-value projection을 다시 cache에 저장합니다. 이는 MLP block에서 $O(n)$ algorithm입니다." %}

KV cache로 sampling함으로써, 이전 token을 재처리하지 않으므로 $n$개 token을 생성하는 시간 복잡도를 FFW에서 $$O(n)$$, attention에서 $$O(n^2)$$로 줄였습니다. 하지만 sequence를 생성하기 위해서는 여전히 많은 forward pass가 필요합니다. Gemini나 ChatGPT를 쿼리할 때 결과가 스트리밍되는 것이 바로 이 때문입니다. 모든 token은 (보통) 대규모 모델에 대한 별도의 (하지만 부분적으로 cached된) Transformer 호출입니다.

곧 보겠지만 <b style="color: red;">prefill</b>과 <b style="color: blue;">generation</b>은 매우 다른 특성을 가집니다 —— Transformer inference는 두 개의 task가 위장한 것입니다! Training과 비교할 때, KV cache도 새롭고 중요한 복잡성의 원천입니다.

### 실제로 우리가 최적화하고자 하는 것은?

더 나아가기 전에, inference에서 완전히 새로운 측면인 latency를 강조할 가치가 있습니다. Training 중에는 throughput(초당 처리된 총 token 수)에만 관심을 두는 반면, inference 중에는 token을 얼마나 빨리 생성하는지(**Time To First Token (TTFT)**과 **per-token latency** 모두)를 걱정해야 합니다. 예를 들어:

* **Offline batch inference**는 eval과 data generation을 위해 inference의 bulk cost에만 관심을 두고 개별 sample의 latency는 무시합니다.
* **Chat interface/streaming task**는 낮은 TTFT를 가지고 인간 읽기 속도를 초과할 만큼 빠르게 token을 생성하면서 대규모로 저렴하게 실행되어야 합니다.
* **Edge inference**(예: 노트북의 `llama.cpp`)는 잠재적으로 심각한 하드웨어 제약 하에서 가능한 한 낮은 latency로 한 번에 한 사용자만 서비스하면 됩니다.

하드웨어 활용도 최대화는 여전히 중요하며 cost와 TTFT에 도움이 되지만, training과 달리 모든 상황에서 개별 사용자의 더 나은 경험으로 *반드시* 변환되지는 않습니다. accelerator, 시스템 및 모델 아키텍처 수준의 많은 최적화는 latency, throughput, context length, 심지어 모델 품질 간의 tradeoff를 만듭니다.

### Transformer의 더 세분화된 관점

지금까지 우리는 대부분 Transformer를 feedforward block의 stack으로 취급했습니다. 이는 FLOP과 memory 관점에서 종종 합리적이지만, inference를 적절히 모델링하기에는 충분하지 않습니다.<d-footnote>이 섹션 전체에서 알게 될 한 가지는 inference가 training보다 훨씬 덜 관대하다는 것입니다. 일반적으로 FLOP이 훨씬 적고, batching 기회가 적으며, latency에 대한 민감도가 훨씬 큽니다. KV cache도 inference를 극적으로 복잡하게 만듭니다.</d-footnote> [Part 4](../transformers-kr)에서 본 바와 같이, Transformer forward pass의 주요 구성 요소는:

1. **많은 linear operation**: MLP($$W_{in}$$, $$W_{out}$$)와 attention QKV projection 및 output projection($$W_Q$$, $$W_K$$, $$W_V$$, $$W_O$$)을 포함합니다. 이들은 모두 HBM에서 parameter와 activation batch를 읽고, 일부 FLOP을 수행하고, 결과를 다시 HBM에 쓰는 것과 관련됩니다.
2. **Dot-product attention**. key-value projection batch와 query activation batch를 HBM에서 읽고, 몇 가지 inner product와 일부 softmax operation을 수행하고, attention 결과를 다시 HBM에 써야 합니다.
3. **그 외 모든 것**: layer norm 적용, activation function, token sampling, KV cache 업데이트, positional embedding을 포함합니다. 이들은 일부 FLOP을 사용하지만 위의 것들에 의해 지배되거나 그들과 함께 융합됩니다.

다음 몇 섹션에서는 prefill과 generation의 맥락에서 이들 각각을 살펴보고 성능을 bottleneck할 가능성이 있는 것이 무엇인지 물어볼 것입니다. 단일 accelerator 내에서 우리는 compute-bound인가 memory-bound인가? prefill 대 generation에 대한 답이 얼마나 다를지 강조하고 싶습니다.

### Linear operation: 무엇이 bottleneck인가?

모든 linear operation은 MLP block에 있든 attention에 있든 개념적으로 동일합니다. 이들의 arithmetic intensity는 batch size에 따라 달라집니다. [Section 1](../roofline-kr)에서 이 수학을 했지만 반복할 가치가 있습니다. $$\text{bf16[B, D]}$$ batch를 $$\text{bf16[D, F]}$$ matrix로 곱하는 단일 matrix multiplication을 살펴보겠습니다. 이는 큰 MLP block이나 더 작은 attention projection($$W_Q$$, $$W_K$$, $$W_V$$, $$W_O$$) 중 하나일 수 있습니다. 이 matrix multiplication을 수행하려면, 이 두 배열을 HBM에서 MXU로 로드하고, multiplication을 수행한 다음, 결과를 다시 HBM에 써야 합니다. 이전과 같이, 우리는:

$$T_\text{math} = \frac{\text{Total FLOPs}}{\text{TPU FLOPs/s}} = \frac{2BDF}{\text{TPU FLOPs/s}}$$

$$T_\text{comms} = \frac{\text{Total Bytes}}{\text{HBM Bandwidth}} = \frac{2BD + 2FD + 2BF}{\text{HBM Bandwidth}}$$

TPU는 compute를 수행하면서 loading을 overlap할 수 있으므로, compute-bound가 되려면 $$T_\text{math} \geq T_\text{comms}$$가 필요합니다:

$$\frac{2BDF}{2BD + 2DF + 2BF} \geq \frac{\text{TPU FLOPs/s}}{\text{HBM Bandwidth}} = \frac{1.97E+14}{8.20E+11} = 240$$

여기서 RHS는 하드웨어의 arithmetic intensity입니다. 이제 $$D$$와 $$F$$가 $$B$$에 비해 매우 크다고 가정하면(보통 batch는 최대 500이고 $$D$$와 $$F > 10k$$), $$\small{2BD + 2DF + 2BF \approxeq 2DF}$$라는 사실을 사용하여 분모를 단순화할 수 있습니다:

$$\begin{align*}
\frac{2BDF}{2BD + 2DF + BF} \approxeq \frac{2BDF}{2DF} \geq \frac{\text{TPU FLOPs/s}}{\text{HBM Bandwidth}} \\
= \frac{1.97E+14}{8.20E+11} \implies B \geq 240 = B_{\text{crit}}
\end{align*}$$

<p markdown=1 class="takeaway">**핵심 내용:** 모든 matrix multiplication에서 compute-bound가 되려면, 총 token batch size가 하드웨어와 quantization에 따라 달라지는 $$B_\text{crit}$$보다 커야 합니다. TPU v5e에서 bf16 activation의 경우 이는 240 token입니다. 이는 Transformer의 모든 간단한 matmul(예: MLP block이나 attention projection)에 적용됩니다.</p>

Training 중에는 매우 큰 batch에 걸쳐 동일한 weight를 재사용하기 때문에 모든 matrix multiplication에서 높은 intensity를 가질 것입니다. **그 높은 arithmetic intensity는 사용자 prompt가 일반적으로 수백에서 수천 token이 길기 때문에 prefill로 이어집니다.** 이전에 본 바와 같이, TPUv5e의 하드웨어 arithmetic intensity는 240이므로, 240 token보다 긴 sequence가 bf16에서 실행되는 dense model에 입력되면 compute-bound가 될 것으로 예상되며 모든 것이 잘 됩니다. 이보다 짧은 prompt는 기술적으로 함께 batching하여 더 높은 활용도를 달성할 수 있지만, 일반적으로 필요하지 않습니다.

<p markdown=1 class="takeaway">**핵심 내용:** prefill 중에는 모든 matrix multiplication이 기본적으로 항상 compute-bound입니다. 따라서 단순히 하드웨어 활용도나 MFU(Model FLOPs Utilization)를 최대화하는 것만으로도 chip당 throughput(cost)과 latency(TTFT 형태)를 최대화하기에 충분합니다. prompt가 극도로 짧지 않는 한, per-prompt 수준에서의 batching은 prefill throughput의 작은 개선을 위해 latency만 추가합니다.</p>

하지만 generation 중에는 단계 간에 순차적 의존성이 있기 때문에 각 요청에 대해 한 번에 한 token씩만 forward pass를 수행할 수 있습니다! 따라서 batch dimension에 걸쳐 병렬화하여 여러 요청을 함께 batching함으로써만 (쉽게) 좋은 활용도를 달성할 수 있습니다. 이에 대해서는 나중에 더 이야기하겠지만, 실제로 latency에 영향을 주지 않으면서 많은 동시 요청을 함께 batching하는 것은 어렵습니다. 그런 이유로 **generation으로 하드웨어 FLOP을 포화시키는 것이 훨씬 어렵습니다.**

<p markdown=1 class="takeaway">**핵심 내용:** generation이 linear/feed-forward operation에서 compute-bound가 되려면 총 token batch size가 $$B_{\text{crit}}$$보다 커야 합니다(TPU v5e에서 bf16 param의 경우 240). generation이 token-by-token으로 순차적으로 발생하기 때문에, 이는 여러 요청을 함께 batching해야 하는데, 이는 어렵습니다!</p>

*이것이 얼마나 큰지 주목할 가치가 있습니다!* Generate batch size 240은 240개의 동시 요청이 한 번에 생성되고 있다는 것을 의미하며, dense model의 경우 240개의 별도 KV cache를 의미합니다. 이는 일부 bulk inference 설정을 제외하고는 실제로 달성하기 어렵다는 것을 의미합니다. 반면, prefill 중에 240개 이상의 token을 밀어넣는 것은 꽤 일상적이지만, sparsity가 증가할수록 일부 주의가 필요합니다.

**이 정확한 수는 quantization과 하드웨어의 종류에 따라 다를 것입니다.** Accelerator는 종종 더 낮은 정밀도에서 더 많은 arithmetic을 제공할 수 있습니다. 예를 들어, int8 parameter를 가지고 있지만 bf16에서 계산을 수행한다면, critical batch size는 120으로 떨어집니다. int8 activation과 int8 param을 사용하면, TPUv5e가 400 TOPs/s의 int8 x int8을 제공할 수 있기 때문에 다시 240으로 뛰어오릅니다.

### Attention은 어떤가?

KV cache를 고려해야 하므로 dot-product attention operation을 살펴보면 상황이 더 복잡해집니다. pure multi-headed attention을 사용하는 단일 attention head만 살펴보겠습니다. 단일 Flash Attention fusion에서<d-footnote>softmax, mask 등을 적용하는 non-matmul FLOP을 무시하여 상당히 단순화하고 있습니다. 이들은 computation이나 HBM read와 overlap되어야 하지만, 특정 TPU generation에서는 non-trivial할 수 있습니다. 이러한 세부사항은 주요 메시지를 바꾸지 않는데, 이는 KV cache가 보통 memory bound라는 것입니다.</d-footnote>:

1. HBM에서 $$\text{bf16[B, T, D]}$$ 형태의 $$Q$$ activation을 읽습니다.
2. HBM에서 $$\text{bf16[B, S, D]}$$ tensor 쌍인 KV cache를 읽습니다.
3. $$QK$$ matmul에서 $$2BSTD$$ FLOP을 수행합니다. Flash Attention을 사용하면 $$\text{bf16[B, S, T]}$$ attention matrix를 다시 HBM에 쓸 필요가 없습니다.
4. attention $$AV$$ matmul에서 $$2BSTD$$를 수행합니다.  
5. 결과 $$\text{bf16[B, T, D]}$$ tensor를 다시 HBM에 씁니다.

모든 것을 합치면:

$$\text{Multiheaded Attention Arithmetic Intensity} = \frac{4BSTD}{4BSD + 4BTD} = \frac{ST}{S+T}$$

Prefill의 경우, self-attention을 수행하므로 $$S=T$$이며, 이는 $$T^2 / 2T = T / 2$$로 단순화됩니다. 이는 **prefill 중 attention의 arithmetic intensity가 $$\Theta(T)$$**라는 것을 의미하므로 훌륭합니다. 이는 attention에서 compute-bound가 되기 꽤 쉽다는 것을 의미합니다. sequence length가 상당히 크기만 하면 괜찮을 것입니다!

하지만 generation은 trivial한 sequence dim을 가지고 있고, $$B$$와 $$D$$ dim이 상쇄되므로, 다음과 같이 근사할 수 있습니다:

$$S \gg T = 1 \implies \frac{ST}{S+T} \approx 1$$

이는 generation 중 attention의 arithmetic intensity를 개선하기 위해 할 수 있는 것이 없다는 것을 의미하므로 나쁩니다. 거대한 KV cache를 로드하면서 아주 적은 양의 FLOP을 수행하고 있습니다. **따라서 우리는 attention 중에 기본적으로 항상 memory bandwidth-bound입니다!**

<p markdown=1 class="takeaway">**핵심 내용:** prefill 중에는 합리적인 sequence length(대략 $$\gt 480$$ token)에 대해 attention이 보통 compute bound인 반면, generation 중에는 arithmetic intensity가 낮고 일정하므로 항상 memory bandwidth-bound입니다.</p>

*이것이 개념적으로 왜 그런가?* 주로, parameter(memory bandwidth가 많이 필요한 구성 요소)가 많은 batch item에 대해 재사용되기 때문에 모델의 linear 부분에서는 compute-bound입니다. 하지만 모든 batch item은 자체 KV cache를 가지므로, batch size가 클수록 더 많은 KV cache를 의미합니다. 아키텍처가 적극적으로 조정되지 않는 한 거의 *항상* memory bound가 될 것입니다.

이는 또한 param memory가 KV cache memory와 비슷해지면 batch size 증가로부터 throughput에서 수익 감소를 얻게 될 것임을 의미합니다. 수익 감소가 얼마나 해로운지는 단일 sequence에 대한 parameter 대 KV cache byte의 비율, 즉 대략 $$2DF / SHK$$의 비율에 달려 있습니다. $$HK\approx D$$이므로, 이는 대략 $$F$$ 대 $$S$$, sequence length의 비율에 달려 있습니다. 이는 또한 KV cache를 더 작게 만드는 아키텍처 수정에 달려 있습니다(곧 더 자세히 설명하겠습니다).

### LLM latency와 throughput의 이론적 추정

이 수학으로부터 최적화할 때 목표해야 할 step time에 대한 꽤 좋은 bound를 얻을 수 있습니다. **(참고: 이 전체 챕터에서 독자가 얻어갔으면 하는 한 가지가 있다면, 그것은 다음 내용입니다).** Generation 중 작은 batch size의 경우(이는 일반적입니다), attention과 MLP block 모두에서 memory bandwidth bound라고 가정하여 per-step latency의 하한을 구할 수 있습니다:

$$\begin{equation*}
\text{Theoretical Min Step Time} = \frac{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}{\text{Total Memory Bandwidth}}
\end{equation*}$$

마찬가지로 throughput의 경우:

$$\begin{equation*}
\text{Theoretical Max Tokens/s} = \frac{\text{Batch Size} \times \text{Total Memory Bandwidth}}{\text{Batch Size} \times \text{KV Cache Size} + \text{Parameter Size}}
\end{equation*}$$

결국 batch size가 커지면 FLOP이 parameter loading을 지배하기 시작하므로, 실제로는 더 일반적인 방정식을 갖습니다:

$$\begin{align}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align}$$

여기서 attention 구성 요소(왼쪽)는 절대 compute-bound가 되지 않으므로 FLOP roofline이 필요하지 않습니다. 이들은 대략적인 계산에 상당히 유용합니다. 예를 들어:

<b markdown=1 style="color: #57cf57;">Pop Quiz:</b> TPU v5e 4x4 slice에서 int8로 30B parameter dense model로부터 batch size 4 token으로 generate step을 수행하고 싶다고 가정하고, bf16 FLOP, 8192 context 및 100 kB / token KV cache를 사용한다고 하겠습니다. 이 operation의 latency에 대한 합리적인 하한은 무엇인가요? 256 token batch를 sampling하고 싶다면 어떨까요?

{% details 답을 보려면 클릭하세요. %}

**답:** int8에서 parameter는 30e9 byte를 사용하고 주어진 spec으로 KV cache는 각각 `100e3 * 8192 = 819MB`를 사용할 것입니다. 16개의 chip이 있고, 각각 `8.1e11` bytes/s의 bandwidth와 `1.97e14` bf16 FLOPs/s를 가집니다. 위 방정식에서, 작은 batch size를 가지고 있으므로 step time이 최소 `(4 * 819e6 + 30e9) / (16 * 8.1e11) = 2.5 ms`일 것으로 예상합니다. 256 token에서는 MLP block에 대해 compute-bound regime에 충분히 들어갈 것이므로 step time은 대략 `(256 * 819e6) / (16 * 8.1e11) + (2 * 256 * 30e9) / (16 * 1.97e14) = 21ms`입니다.

{% enddetails %}

보시다시피, 여기에는 throughput과 latency 간의 명확한 tradeoff가 있습니다. 작은 batch는 빠르지만 하드웨어를 잘 활용하지 못합니다. 큰 batch는 느리지만 효율적입니다. 다음은 일부 이전 PaLM model에 대해 계산된 latency-throughput Pareto frontier입니다([ESTI paper](https://arxiv.org/pdf/2211.05102)<d-cite key="esti"></d-cite>에서):

{% include figure.liquid path="assets/img/latency-cost.png" class="img-fluid" caption="<b>그림:</b> 여러 PaLM model에 대한 cost(즉, throughput) 대 latency의 Pareto frontier. chip count(C)와 batch size(B)가 Pareto frontier를 따라 어떻게 이동시키는지 주목하세요. 단, green dot(PaLM 540B의 C:32 B:16)은 예외로, 사용 가능한 memory가 좋은 batch size를 지원하는 것을 방해하여 throughput이 저하되었습니다. batch size 240 이후에 throughput이 일반적으로 어떻게 평평해지는지 주목하세요. int8 weight는 더 나은 latency-throughput pareto optimal을 제공하지만, 더 나은 최대 throughput을 제공하지는 않습니다." %}

batch size를 knob으로 하여 latency와 throughput을 tradeoff할 뿐만 아니라, HBM에 의해 제한되는 경우 더 큰 batch를 맞출 수 있도록 더 작은 topology보다는 더 큰 topology를 선호할 수도 있습니다. [다음 섹션](../applied-inference-kr)에서 이를 더 자세히 탐구합니다.

<p markdown=1 class="takeaway">**핵심 내용:** generation throughput에 관심이 있다면, 가능한 한 가장 큰 per-chip batch size를 사용하십시오. TPU arithmetic intensity($$B_\text{crit}$$, 보통 120 또는 240) 이상의 모든 per-chip batch size는 throughput을 최대화할 것입니다. 이를 달성하기 위해 topology를 늘려야 할 수도 있습니다. 더 작은 batch size는 throughput 비용으로 latency를 개선할 수 있게 해줄 것입니다.</p>

{% details 하드웨어 관점에서 몇 가지 주의사항이 있습니다. 몇 가지 세부사항을 보려면 클릭하세요. %}

이것은 모두 꽤 이론적입니다. 실제로는 몇 가지 이유로 sharp roofline을 완전히 보지 못하는 경우가 많습니다:

* HBM read가 FLOP과 완벽하게 overlap될 것이라는 가정은 컴파일러(XLA)가 오류가 있기 때문에 현실적이지 않습니다.
* Sharded model의 경우, XLA는 또한 종종 model-sharded matrix multiple의 ICI communication을 FLOP 자체와 효율적으로 overlap하지 못하므로, $$\text{BS}=32$$를 넘어서면 linear에서 latency hit를 받기 시작하는 경우가 많습니다.
* 이론적 roofline보다 큰 batch size는 불완전한 overlapping 때문에 여전히 throughput에서 일부 개선을 볼 것이지만, 이는 좋은 heuristic입니다.

{% enddetails %}

### Memory는 어떤가?

bandwidth와 FLOP을 살펴보는 데 시간을 보냈지만 memory는 살펴보지 않았습니다. 새로운 data structure인 KV cache 덕분에 inference time에 memory 상황은 매우 다르게 보입니다. 이 섹션에서는 상황이 얼마나 다르게 보이는지 보여주기 위해 실제 모델(LLaMA 2-13B)을 선택하겠습니다:

| hyperparam               | value  |
| ------------------------ | ------ |
| n\_layers (L)            | 40     |
| d\_model (D)             | 5,120  |
| ffw\_multiplier (F // D) | 2.7    |
| n\_heads (N)             | 40     |
| n\_kv\_heads (K)         | 40     |
| d\_qkv (H)               | 128    |
| n\_embeddings (V)        | 32,000 |

Inference 중에 무엇이 memory를 사용하고 있나요? 분명히 parameter입니다. 이들을 세어보면:

| param            | formula                                                                                                                   | size (in bytes)                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| FFW params       | d_model<sup>2</sup> x ffw\_multiplier x 3 (for gelu \+ out-projection) x n\_layers                                        | 5,120 x 5,120 x 2.7 x 3 x 40 \= **8.5e9**                         |
| Vocab params     | 2 (input and output embeddings) x n\_embeddings x d\_model                                                                | 2 x 32,000 x 5,120 \= **0.3e9**                                   |
| Attention params | \[2 (*q and output*) x d\_model x n\_heads x d\_qkv \+ 2 (*for k and v*) x d\_model x n\_kv\_heads x d\_qkv\] x n\_layers | (2 x 5,120 x 40 x 128 \+ 2 x 5,120 x 40 x 128\) x 40 \= **4.2e9** |

이 parameter들을 합하면 8.5e9 + 4.2e9 + 0.3e9 = **13e9 total parameter**가 예상대로 나옵니다. 이전 섹션에서 본 바와 같이, training 중에는 parameter를 bfloat16에 float32의 optimizer state와 함께 저장할 수 있습니다. 이는 약 100GB의 memory를 사용할 수 있습니다. 이는 여러 TB를 사용할 수 있는 gradient checkpoint와 비교하면 초라합니다.

**Inference는 어떻게 다른가요?** Inference 중에는 parameter의 한 copy를 저장하는데, bfloat16이라고 하겠습니다. 이는 26GB를 사용합니다 — 실제로는 quantization을 통해 이보다 훨씬 나을 수 있는 경우가 많습니다. 추적할 optimizer state나 gradient가 없습니다. checkpoint를 하지 않기 때문에(backward pass를 위해 activation을 보관하지 않음), 우리의 activation footprint는 prefill<d-footnote>특히 attention matrix를 구체화하는 것을 피하는 Flash Attention 덕분입니다</d-footnote>과 generate 모두에서 무시할 수 있습니다. 8k token을 prefill한다면, 단일 activation은 약 `8,192 x 5,120 x 2 bytes = 80MB`의 memory만 사용합니다. 더 긴 prefill은 많은 더 작은 forward pass로 나눌 수 있으므로, 더 긴 context에서도 문제가 되지 않습니다. Generation은 그보다 더 적은 token을 사용하므로 activation은 무시할 수 있습니다.

**주요 차이는 KV cache입니다**. 이들은 모든 과거 token에 대한 key와 value projection으로, 허용되는 최대 sequence length에 의해서만 크기가 제한됩니다. $$T$$ token에 대한 총 크기는

$$\text{KV cache size} = 2 \cdot \text{bytes per float} \cdot H \cdot K \cdot L \cdot T$$

여기서 $$H$$는 각 head의 dimension, $$K$$는 KV head 수, $$L$$은 layer 수이고, 2는 key와 value 모두를 저장하는 것에서 나옵니다.

**이는 적당한 batch size와 context length에서도 매우 빠르게 커질 수 있습니다**. LLaMA-13B의 경우, bf16에서 단일 8192 sequence에 대한 KV cache는

$$8192\ (T) \times 40\ (K) \times 128\ (H) \times 40\ (L) \times 2\ (\text{bytes}) \times 2 = 6.7 \text{GB}$$

**이 중 4개만으로도 parameter 사용량을 초과합니다!** 명확히 하자면, LLaMA 2는 더 긴 context에서 KV cache size에 대해 최적화되지 않았습니다(보통 $$K$$가 LLaMA-3에서처럼 훨씬 작기 때문에 항상 이렇게 나쁘지는 않습니다). 하지만 여전히 시사하는 바가 있습니다. memory나 latency 추정에서 이들을 무시할 수 없습니다.

### LLaMA 2-13B의 throughput과 latency 모델링

8xTPU v5e에서 앞서 도출한 최대 이론적 throughput을 위한 critical batch size(240)까지 다양한 batch size에서 generation을 완벽하게 효율적으로 수행하려고 하면 어떻게 될지 살펴보겠습니다.

| Batch Size                        |      1 |      8 |     16 |     32 |     64 |    240 |
| :-------------------------------- | -----: | -----: | -----: | -----: | -----: | -----: |
| KV Cache Memory (GiB)             |    6.7 |   53.6 |  107.2 |  214.4 |  428.8 |   1608 |
| Total Memory (GiB)                |   32.7 |   79.6 |  133.2 |  240.4 |  454.8 |   1634 |
| Theoretical Step Time (ms)        |   4.98 |  12.13 |  20.30 |  36.65 |  69.33 | 249.09 |
| Theoretical Throughput (tokens/s) | 200.61 | 659.30 | 787.99 | 873.21 | 923.13 | 963.53 |

8x TPU v5e는 128GiB의 HBM, 6.5TiB/s의 HBM bandwidth(각각 0.82TiB/s), 1600TF/s의 compute를 제공합니다.

이 모델의 경우, batch size를 늘리면 더 나은 throughput을 얻지만 빠르게 수익이 감소합니다. batch size 16을 넘어서면 OOM이 발생하고, 240에 가까워지려면 한 자릿수 많은 memory가 필요합니다. 더 큰 topology는 latency를 개선할 수 있지만, per chip throughput에는 벽에 부딪혔습니다.

같은 수의 총 param을 유지하면서 KV cache를 마술적으로 5배 작게 만든다고 해보겠습니다(예: 1:5 [GMQA](#tricks-for-improving-generation-throughput-and-latency)를 사용하여, 40개 Q head에 걸쳐 공유되는 8개 KV head를 가지는 것 — 자세한 내용은 다음 섹션 참조).

| Batch Size                        |      1 |        8 |       16 |       32 |       64 |      240 |
| :-------------------------------- | -----: | -------: | -------: | -------: | -------: | -------: |
| KV Cache Memory (GiB)             |   1.34 |    10.72 |    21.44 |    42.88 |    85.76 |    321.6 |
| Total Memory (GiB)                |  27.34 |    36.72 |    47.44 |    68.88 |   111.76 |    347.6 |
| Theoretical Step Time (ms)        |   4.17 |     5.60 |     7.23 |    10.50 |    17.04 |    52.99 |
| Theoretical Throughput (tokens/s) | 239.94 | 1,429.19 | 2,212.48 | 3,047.62 | 3,756.62 | 4,529.34 |

더 작은 KV cache로도 여전히 수익 감소가 있지만, per chip의 이론적 throughput은 batch size 240까지 계속 확장됩니다. 훨씬 더 큰 64 batch를 맞출 수 있고, latency도 모든 batch size에서 일관되게 더 좋습니다. Latency, 최대 throughput, 최대 batch size가 모두 극적으로 개선됩니다! 실제로 후속 LLaMA generation들은 바로 이 최적화를 사용했습니다 — LLaMA-3 8B는 32개 query head와 8개 KV head를 가집니다([출처](https://huggingface.co/MaziyarPanahi/Llama-3-13B-Instruct-v0.1/blob/dfdeb40bdb2c149dfa399ea2be0d56eb120f0831/config.json)).

<p markdown=1 class="takeaway">**핵심 내용:** Parameter 외에도 KV cache의 크기가 모델의 궁극적인 inference 성능에 많은 영향을 미칩니다. 아키텍처 결정과 runtime 최적화의 조합으로 이를 통제하에 두고 싶어합니다.</p>

## Generation Throughput과 Latency 개선 기법

원래 [Attention is All You Need paper](https://arxiv.org/abs/1706.03762) 이후로, 모델을 더 효율적으로 만드는 많은 기법이 개발되었으며, 종종 특히 KV cache를 대상으로 합니다. 일반적으로 말하면, 더 작은 KV cache는 latency에 해를 끼치지 않으면서 generation step의 batch size와 context length를 늘리기 쉽게 만들고, Transformer를 둘러싼 시스템(request caching 같은)의 삶을 더 쉽게 만듭니다. 품질에 대한 영향을 무시하면:

**Grouped multi-query attention (aka GMQA, GQA):** KV head의 수를 줄이고, attention mechanism에서 많은 Q head와 공유할 수 있습니다. 극단적인 경우, 모든 Q head에 걸쳐 단일 KV head를 공유하는 것이 가능합니다. 이는 pure MHA에 비해 Q:KV 비율의 배수만큼 KV cache를 줄이며, 모델의 성능이 이 변화에 상대적으로 민감하지 않다는 것이 관찰되었습니다.

{% include figure.liquid path="assets/img/gmqa.png" class="img-fluid" %}

이는 또한 attention computation의 arithmetic intensity를 효과적으로 증가시킵니다([Section 4](../transformers-kr)의 Question 4 참조).

**일부 local attention layer 혼합:** Local attention은 context를 작은 것에서 적당한 크기의 최대 길이로 제한합니다. Training time과 prefill time에 이는 attention matrix를 삼각형 대신 대각선 strip으로 masking하는 것을 포함합니다. 이는 local layer에 대한 KV cache의 최대 길이를 효과적으로 제한합니다. 일부 local layer를 일부 global layer와 함께 모델에 혼합함으로써, local window보다 긴 context에서 KV cache size가 크게 줄어듭니다.

**Layer 간 KV 공유:** 모델은 일부 패턴으로 layer 간에 동일한 KV cache를 공유하는 것을 학습할 수 있습니다. 이는 KV cache size를 줄이고, batch size 증가, caching, offline storage 등에서 이익을 제공하지만, 공유된 KV cache는 HBM에서 여러 번 읽어야 할 수 있으므로 *반드시 step time을 개선하지는 않습니다.*

{% include figure.liquid path="assets/img/kv-sharing.png" class="img-fluid" caption="
 <b>Left:</b> Multiple layers of pure global attention. <b>Right:</b> An example of some global/local interleaving pattern with sharing with adjacent layers. 출처: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai blog</a>."%}

**Quantization:** Inference는 보통 parameter와 KV의 정밀도에 덜 민감합니다. Parameter와 KV cache를 quantization(예: int8, int4, `fp8` 등)함으로써 두 경우 모두에서 memory bandwidth를 절약하고, compute roofline에 도달하는 데 필요한 batch size를 줄이고, 더 큰 batch size에서 실행할 memory를 절약할 수 있습니다. Quantization은 모델이 quantization으로 training되지 않았더라도 종종 post training에 적용할 수 있다는 추가 장점이 있습니다.

**Ragged HBM read와 Paged Attention 사용:** 위 계산에서 각 KV cache에 대해 8k의 context를 할당했지만 전체 KV cache를 memory에서 읽는 것이 종종 필요하지 않습니다 — 요청은 광범위한 길이 분포를 가지며 모델의 최대 context를 사용하지 않으므로, KV cache의 non-padding 부분만 읽는 kernel(예: Flash Attention variant)을 종종 구현할 수 있습니다.

Paged Attention<d-cite key="paged"></d-cite>은 KV cache를 OS-style page table에 저장하고 KV cache의 padding을 대부분 피하는 이에 대한 개선입니다. 이는 많은 복잡성을 추가하지만 모든 batch가 필요한 만큼의 memory만 사용한다는 것을 의미합니다. 이는 runtime 최적화이므로 아키텍처에 무관합니다.

{% include figure.liquid path="assets/img/paged-attention.png" class="img-fluid img-small" caption="<b>그림:</b> generation 중에 단일 token(fourth)이 여러 KV cache block/page에 attend합니다. KV cache를 paging함으로써 필요한 것보다 더 많은 memory를 로드하거나 저장하는 것을 피합니다. <a href=\"https://arxiv.org/pdf/2309.06180\">PagedAttention paper</a>에서 가져옴." %}

<p markdown=1 class="takeaway">**큰 그림:** 전체적으로, 이러한 KV cache 최적화는 표준 MHA Transformer에 비해 KV cache size를 한 자릿수 이상 줄일 수 있습니다. 이는 Transformer의 전체 cost에서 한 자릿수 개선으로 이어질 수 있습니다.</p>

## 여러 Accelerator에 걸친 Inference 분산

지금까지 단일 chip을 넘어 어떻게 확장하는지에 대해서는 두루뭉술하게 넘어갔습니다. [Section 5](../training-kr)를 따라, 우리가 사용할 수 있는 다양한 전략과 그 tradeoff를 탐구해보겠습니다. 항상 그렇듯이, prefill과 generation을 별도로 살펴보겠습니다.

### Prefill

Roofline 관점에서, **prefill은 training과 거의 동일하며** 거의 모든 동일한 기법과 tradeoff가 적용됩니다 — model (Megatron) parallelism, sequence sharding(충분히 긴 context에 대해), pipelining, 심지어 FSDP까지 모두 가능합니다! 나중에 generation을 할 수 있도록 KV를 유지해야 할 뿐입니다. Training에서와 같이, chip 수를 늘리면 더 많은 FLOPs/s에 접근할 수 있지만(잠재적으로 더 낮은 TTFT), communication overhead가 추가됩니다(잠재적으로 chip당 throughput이 감소).

**Prefill sharding을 위한 일반적인 규칙:** 다음은 prefill을 위한 일반적인 규칙 set입니다. 단일 sequence에서만 prefill을 수행한다고 가정하겠습니다(batch dimension 없음):

1. *Model sharding:* 일반적으로 ICI-bound가 될 때까지 어느 정도의 model parallelism을 먼저 수행합니다. [Section 5](../training-kr)에서 본 바와 같이, 이는 1 axis에 대해 약 $$F / 2550$$ 정도입니다(보통 4-8 way sharding 정도).
2. *Sequence parallelism:* 이를 넘어서는 sequence parallelism을 수행합니다(data parallelism과 같지만 sequence dimension에 걸쳐 sharding). Sequence parallelism은 attention에서 일부 extra communication을 도입하지만, 더 긴 context에서는 일반적으로 상당히 작습니다. Training에서와 같이, communication과 computation을 overlap할 수 있습니다(각각 Megatron과 ring attention에 대해 collective matmul 사용).

<p markdown=1 class="takeaway">**핵심 내용:** prefill 중에는 training 중에 작동할 수 있는 거의 모든 sharding이 잘 작동할 수 있습니다. ICI bound까지 model parallelism을 하고, 그 다음 sequence parallelism을 하십시오.</p>

### Generation

Generation은 prefill보다 더 복잡한 존재입니다. 우선, 많은 요청을 함께 batch해야 하기 때문에 큰 batch size를 얻기가 더 어렵습니다. Latency target이 더 낮습니다. 이들을 합치면 일반적으로 더 memory-bound이고 communication overhead에 더 민감하다는 것을 의미하며, 이는 sharding 전략을 제한합니다:

1. **FSDP는 불가능:** HBM에서 MXU로 parameter와 KV cache를 로드하는 데 memory-bound이므로, HBM보다 몇 자릿수 느린 ICI를 통해 이들을 이동시키고 싶지 않습니다. *weight가 아닌 activation을 이동시키고 싶습니다.* 이는 FSDP와 유사한 방법이 generation에서 보통 완전히 불가능하다는 것을 의미합니다.<d-footnote>Training 후에 실수로 켜둔 채로 두는 것은 한 자릿수 regression을 일으키는 쉽고 일반적인 방법입니다</d-footnote>

2. **Data parallelism을 할 이유가 없음:** Pure data parallelism은 parameter를 복제하고 parameter를 더 빨리 로드하는 데 도움이 되지 않기 때문에 도움이 되지 않습니다. 대신 모델의 여러 copy를 돌리는 것이 낫습니다.<d-footnote>이는 더 작은 batch size에서 모델의 copy로 여러 서버를 돌린다는 의미입니다. 모델 수준의 data parallelism은 엄격히 더 나쁩니다.</d-footnote>

3. **Sequence가 없으면 sequence sharding도 없음.** Sequence sharding에 행운을 빌어요.

_이는 대부분 dense model generation에 대해 model sharding의 variant를 남겨둡니다_. Prefill에서와 같이, 우리가 할 수 있는 가장 간단한 것은 간단한 model parallelism(activation이 완전히 복제되고, weight가 MLP의 hidden dimension에 걸쳐 완전히 sharded)을 ICI bound가 될 때까지 4-8 way까지 하는 것입니다. 하지만 종종 memory bandwidth bound이므로, 실제로 이 한계를 넘어서 latency를 개선할 수 있습니다!

**Generation에 대한 ICI bound 참고:** Training 중에는 compute-bound가 되고 싶어서 ICI comm이 FLOP보다 오래 걸릴 때를 보는 roofline을 봅니다. 하지만 generation 중에는 parameter loading에 의해 memory bandwidth bound이면, 이 지점을 넘어서 model sharding을 늘리고 최소한의 throughput cost로 latency를 개선할 수 있습니다. 더 많은 model sharding은 weight를 로드할 더 많은 HBM을 제공하고, FLOP은 중요하지 않습니다.<d-footnote>FLOP time이 우리를 bottleneck하지 않는다는 의미에서, 우리가 걱정해야 할 것은 ICI time이 parameter loading time을 초과하는 것입니다.</d-footnote> bottleneck이 되기 전에 얼마나 많은 model parallelism을 할 수 있는지 살펴보겠습니다.

$$\begin{align*}T_\text{HBM comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{ICI comms} = \frac{2BD}{W_\text{ici}}\end{align*}$$

$$T_\text{ICI comms} > T_\text{HBM comms} \rightarrow \frac{W_\text{hbm}}{W_\text{ici}} > \frac{F}{Y \cdot B} \rightarrow Y > F / (B \cdot \beta)$$

여기서 $$\beta = W_\text{hbm} / W_\text{ici}$$입니다. 이 수는 TPU v5e와 TPU v6e에 대해 보통 8 정도입니다. 즉, 예를 들어 $$F$$가 16,384이고 $$B$$가 32라면, 이론적으로 throughput에서 의미 있는 hit 없이 `16384 / (32 * 8) = 64` way까지 model parallelism을 할 수 있습니다. 이는 KV cache를 64-way로 완전히 shard할 수 있다고 가정하는데, 이는 어렵습니다: 아래에서 이를 논의합니다.

Attention layer의 경우, Megatron style로 head에 걸쳐 attention $$W_Q$$와 $$W_O$$도 model shard합니다. KV weight는 꽤 작고, 이들을 복제하는 것이 $$K$$-way sharding을 넘어 sharding하는 것보다 종종 더 저렴합니다.

<p markdown=1 class="takeaway">**핵심 내용:** generation 중 우리의 유일한 옵션은 model parallelism의 variant입니다. 더 큰 KV cache나 parameter보다는 activation을 이동시키는 것을 목표로 합니다. Batch size가 클 때는 FLOPs-ICI bound($$F / \alpha$$)까지 model parallelism을 합니다. Batch size가 더 작을 때는 (적당한 throughput cost로) 더 많이 model sharding하여 latency를 개선할 수 있습니다. KV head보다 더 많은 way로 model shard하고 싶을 때는 batch dimension에 따라서도 KV를 shard할 수 있습니다.</p>

### KV cache sharding

**Sharding해야 할 추가 data structure — KV cache도 있습니다.** 다시, attention latency의 주요 원천이므로 cache를 복제하는 것을 거의 항상 피하는 것을 선호합니다. 이를 위해 먼저 head dimension을 따라 KV를 Megatron-shard합니다. 이는 $$K$$-way sharding으로 제한되므로, head 수가 적은 모델의 경우 head dimension을 가능한 한 많이 shard하고 그 다음 batch dimension을 따라 shard합니다. 즉, $$\text{KV}[2, B_Z, S, K_Y, H]$$. 이는 KV cache가 완전히 분산된다는 것을 의미합니다.

{% include figure.liquid path="assets/img/esta-figure.png" class="img-fluid" caption="<b>그림:</b> (a) pure model sharding을 가진 Multi head attention과 (b) KV cache의 batch sharding을 가진 Multiquery attention의 attention mechanism 비교. model sharding에서 batch sharding으로 activation을 shift하기 위해 두 개의 extra AllToAll이 필요하므로 KV cache에서 작동할 수 있다는 점에 주목하십시오." %}

이 비용은 모든 attention layer마다 두 개의 AllToAll입니다 — Q activation을 batch sharding으로 shift하여 batch sharding으로 attention을 계산할 수 있도록 하는 것과 batch sharded attention output을 다시 pure model sharded로 shift하는 것입니다.

{% details 전체 algorithm은 여기 있습니다! %}

여기서는 $$Y$$와 $$Z$$ 모두에 걸친 model parallelism을 가진 전체 attention algorithm을 작성하겠습니다. key tensor와 KV head dimension 모두에 $$K$$를 사용해서 죄송합니다. $$M=N/K$$라고 하겠습니다.

<div markdown=1 class="algorithm">

1. X[B, D] = ... (기존 activation, 이전 layer에서 unsharded)
2. K[B<sub>Z</sub>, S, K<sub>Y</sub>, H], V[B<sub>Z</sub>, S, K, H] = ... (기존 KV cache, batch sharded)
3. Q[B, N<sub>YZ</sub>, H] = X[B, D] \* W<sub>Q</sub>[D, N<sub>YZ</sub>, H]
4. Q[B<sub>Z</sub>, N<sub>Y</sub>, H] = **AllToAll**<sub>Z->B</sub>(Q[B, N<sub>YZ</sub>, H])
5. Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = **Reshape**(Q[B<sub>Z</sub>, N<sub>Y</sub>, H])
6. O[B<sub>Z</sub>, S, K<sub>Y</sub>, M] = Q[B<sub>Z</sub>, K<sub>Y</sub>, M, H] \*<sub>H</sub> K[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
7. O[B<sub>Z</sub>, S, K, M] = **Softmax**<sub>S</sub>(O[B<sub>Z</sub>, S, K<sub>Y</sub>])
8. O[B<sub>Z</sub>, K<sub>Y</sub>, M, H] = O[B<sub>Z</sub>, S, K, M] \*<sub>S</sub> V[B<sub>Z</sub>, S, K<sub>Y</sub>, H]
9. O[B, K<sub>Y</sub>, M<sub>Z</sub>, H] = **AllToAll**<sub>Z->M</sub>(O[B<sub>Z</sub>, K<sub>Y</sub>, M, H])
10. O[B, N<sub>YZ</sub>, H] = **Reshape**(O[B, K<sub>Y</sub>, M<sub>Z</sub>, H])
11. X[B, D] {U<sub>YZ</sub>} = W<sub>O</sub>[N<sub>YZ</sub>, H, D] \*<sub>N,H</sub> O[B, N<sub>YZ</sub>, H]
12. X[B, D] = **AllReduce**(X[B, D] { U<sub>YZ</sub>})

이는 꽤 복잡하지만 일반적으로 어떻게 작동하는지 볼 수 있습니다. 새로운 comm은 작은 activation에서 작동하므로 적당히 비싸지만, 그 대가로 KV를 로드하는 데 막대한 양의 memory bandwidth를 절약합니다(이는 stationary).

</div>

{% enddetails %}

* **Sequence sharding:** Batch size가 너무 작거나 context가 길면, KV cache를 sequence shard할 수 있습니다. 다시, 여기서 shard에 걸쳐 attention을 accumulate하는 데 collective cost를 지불합니다. 먼저 Q activation을 AllGather해야 하고, 그 다음 Flash Attention과 유사한 방식으로 KV를 accumulate해야 합니다.

## 효과적인 Inference Engine 설계

지금까지 개별 prefill과 generate operation을 isolation에서 효율적으로 최적화하고 shard하는 방법을 살펴봤습니다. 실제로 효과적으로 사용하려면, latency/throughput Pareto frontier에서 우리가 선택한 지점에서 이 두 operation을 feed할 수 있는 inference engine을 설계해야 합니다.

가장 간단한 방법은 단순히 prefill batch를 실행한 다음 generation batch를 실행하는 것입니다:

{% include figure.liquid path="assets/img/batched-prefill.png" class="img-fluid" caption="<b>그림:</b> 가장 간단한 설정에서는 요청이 집계되고, 서버가 prefill batch를 실행하고 모든 sequence가 완료될 때까지 generate function을 호출하는 것을 번갈아 합니다." %}

이는 구현하기 쉽고 대부분 코드베이스의 첫 번째 inference 설정이지만, 여러 단점이 있습니다:

1. **Latency가 끔찍합니다.** Prefill과 generate batch size를 결합합니다. 큰 prefill batch size에서 Time to first token(TTFT)이 끔찍합니다 — 사용자가 token을 볼 수 있기 전에 모든 prefill을 완료해야 합니다. 작은 batch size에서 generate throughput이 끔찍합니다.
2. **더 짧은 generation을 더 긴 것들이 block합니다.** 많은 sequence가 다른 것들보다 먼저 끝나서 generation 중에 빈 batch slot을 남겨두고 generate throughput을 더욱 해칩니다. Batch size와 generation length가 증가할수록 문제가 악화됩니다.
3. **Prefill이 padding됩니다.** Prefill이 가장 긴 sequence로 padding되어 많은 compute를 낭비합니다. 이에 대한 해결책이 있지만, 역사적으로 XLA는 이러한 FLOP을 건너뛰는 것을 상당히 어렵게 만들었습니다. 다시 batch size와 prefill sequence length가 클수록 이것이 더 나빠집니다.
4. **Prefill과 generation 간에 sharding을 공유해야 합니다.** Prefill과 generate 모두 동일한 slice에 살며, 이는 (weight의 두 copy를 유지하지 않는 한) 둘 다에 대해 동일한 topology와 sharding을 사용한다는 것을 의미하고 일반적으로 성능에 도움이 되지 않습니다. 예를 들어 generate는 훨씬 더 많은 model sharding을 원합니다.

따라서 이 방법은 edge application(보통 단일 사용자 서비스에만 관심이 있고 더 적은 FLOPs/byte를 가진 하드웨어 사용)과 Transformer 코드베이스의 생명주기 초기의 빠른 반복(단순성 때문에)에만 권장됩니다.

약간 더 나은 접근법은 batch size 1에서 prefill을 수행하고(compute-bound이지만 합리적인 latency를 가짐) generation 중에는 여러 요청을 함께 batch하는 것입니다:

{% include figure.liquid path="assets/img/interleaving.png" class="img-fluid" %}

이는 generation throughput을 높게 유지하면서 batched prefill로부터 낭비된 TTFT를 피할 것입니다. Prefill과 generation step을 "interleave"하므로 이를 **interleaved** configuration이라고 부릅니다. 이는 throughput이 주요 목표인 evaluation 같은 bulk generation application에 매우 강력합니다. Orchestrator는 generation slot이 열리는 순간 prefill의 우선순위를 두도록 구성할 수 있어, 매우 큰 generation batch size에 대해서도 높은 활용도를 보장합니다. 또한 다른 요청과 batch되지 않으므로 prefill을 최대 길이로 padding하는 것을 피할 수 있습니다.

주요 단점은 서버가 prefill을 수행할 때 모든 compute resource가 prefill에 의해 소비되므로 다른 모든 요청의 generation이 일시 정지된다는 것입니다. 응답이 바쁘게 decoding되고 있는 사용자 A가 prefill이 발생하고 있는 사용자 B에 의해 block됩니다. 이는 TTFT가 개선되었더라도 token generation이 평균적으로 떨리고 느려질 것임을 의미하며, 이는 많은 application에서 좋은 사용자 경험이 아닙니다 — 다른 사용자의 prefill이 요청의 전체 latency의 critical path에 있습니다.

이를 해결하기 위해 decode와 prefill을 분리합니다. Transformer inference는 하나의 서버에서 수행할 수 있지만, latency 관점에서는 두 가지 다른 task를 두 set의 TPU/GPU에서 실행하는 것이 종종 더 좋습니다. Prefill 서버는 network를 통해 generate 서버로 전송되는 KV cache를 생성하고, generate 서버는 여러 cache를 함께 batch하고 각각에 대해 token을 생성합니다. 이를 **"disaggregated"** serving이라고 부릅니다.

{% include figure.liquid path="assets/img/disaggregation.png" class="img-fluid" %}

이는 몇 가지 장점을 제공합니다:

1. **대규모 저지연:** 충분하지 않은 prefill capacity가 없는 한 사용자의 요청이 다른 사용자의 요청에 의해 block되지 않습니다. 요청은 즉시 prefill되고, generation 서버로 전송되고, 즉시 generation buffer에 슬롯되어야 합니다. 많은 동시 요청이 들어올 것으로 예상되면, 사용자가 확장된 기간 동안 prefill queue에 남겨지지 않도록 generate 서버 수와 독립적으로 prefill 서버 수를 확장할 수 있습니다.

2. **특화:** 종종 prefill과 generate에 대한 latency-optimal parameter sharding 전략/하드웨어 topology가 상당히 다릅니다(예를 들어, 더 많은 model parallelism이 generate에 유용하지만 prefill에는 그렇지 않음). 두 operation을 동일한 sharding을 사용하도록 제약하는 것은 둘 다의 성능을 해치고, 두 set의 weight를 갖는 것은 memory를 사용합니다. 또한 prefill을 자체 서버로 이동시킴으로써 현재 처리 중인 것을 제외한 KV cache를 보유할 필요가 없습니다. 이는 history caching(다음 섹션 참조)이나 prefill latency 최적화를 위해 더 많은 memory를 자유롭게 사용할 수 있다는 것을 의미합니다.

한 가지 단점은 KV cache가 이제 network를 통해 shift되어야 한다는 것입니다. 이는 일반적으로 허용 가능하지만 다시 KV cache size 줄이기에 대한 동기를 제공합니다.

<p markdown=1 class="takeaway">**핵심 내용:** latency에 민감하고 high-throughput serving의 경우, 일반적으로 prefill과 generation을 별도 서버로 분리해야 하며, prefill은 batch 1에서 작동하고 generation은 많은 동시 요청을 함께 batching합니다.</p>

### Continuous Batching

위의 문제 (2)는 **continuous batching**의 개념에 동기를 부여합니다. 다음을 최적화하고 컴파일합니다:

* 가변 context length를 가진 여러 prefill function과 일부 KV buffer에 삽입하는 것, 일부 최대 batch size와 context length/page 수.
* KV cache를 가져와서 현재 활성인 모든 요청에 대해 generation step을 수행하는 generate function.

그 다음 들어오는 요청을 queue하고, 사용 가능한 generate slot에 따라 prefill과 generate를 호출하고, history caching(다음 섹션 참조)을 처리하고 token을 stream out하는 orchestrator와 이러한 function을 결합합니다.

{% include figure.liquid path="assets/img/continuous-batching.gif" class="img-fluid" %}

### Prefix Caching

Prefill은 비싸고 compute-bound이므로(headroom이 적음), 그 cost를 줄이는 가장 좋은 방법 중 하나는 더 적게 하는 것입니다. LLM이 autoregressive이므로, ["I", "like", "dogs"]와 ["I", "like", "cats"] query는 처음 두 token에서 동일한 KV cache를 생성합니다. 이는 원칙적으로 "I like dogs" cache를 먼저 계산하고 그 다음 "I like cats" cache를 계산한다면, 계산의 1/3만 하면 된다는 것을 의미합니다. Cache를 재사용하여 대부분의 작업을 절약할 수 있습니다. 이는 몇 가지 특정 경우에 특히 강력합니다:

1. **Chatbot:** 대부분의 chatbot 대화는 자신에게 엄격히 append하는 back-and-forth dialog를 포함합니다. 이는 각 dialog turn에서 KV cache를 저장할 수 있다면, 가장 새로운 token을 제외한 모든 것에 대한 computation을 건너뛸 수 있다는 것을 의미합니다.
2. **Few-shot prompting:** 어떤 종류의 few-shot prompt가 있다면, 이를 저장하고 무료로 재사용할 수 있습니다. System instruction도 종종 이런 형태를 가집니다.

이것이 어려운 유일한 이유는 memory 제약입니다. 보았듯이, KV cache는 크고(종종 여러 GB), caching이 유용하려면 follow-up query가 도착할 때까지 보관해야 합니다. 일반적으로, prefill 서버에서 사용되지 않는 HBM은 local caching system에 사용할 수 있습니다. 또한 accelerator는 보통 CPU host에 많은 memory를 가지고 있습니다(예: 8xTPUv5e 서버는 128GiB의 HBM을 가지지만 약 450GiB의 Host DRAM을 가짐). 이 memory는 HBM보다 훨씬 느립니다 — 보통 generation step을 하기에는 너무 느립니다 — 하지만 cache read에는 충분히 빠릅니다. 실제로:

* KV cache가 초기 요청을 처리한 TPU set에 local이므로, follow-up query가 동일한 replica에 도착하도록 보장하는 어떤 형태의 affinity routing이 필요합니다. 이는 load balancing에 문제를 일으킬 수 있습니다.
* 더 작은 KV cache가 (다시) 도움이 됩니다 — 같은 공간에 더 많은 KV cache를 저장하고 read time을 줄일 수 있게 해줍니다.
* KV cache와 그 lookup은 tree나 trie에 매우 자연스럽게 저장할 수 있습니다. Eviction은 LRU 기반으로 발생할 수 있습니다.

{% include figure.liquid path="assets/img/prefix-caching-trie.png" class="img-fluid" caption="<b>그림:</b> LRU trie로 구현된 KV prefix cache. prefix를 공유함으로써 KV memory 복제를 피할 수 있습니다. 출처: <a href=\"https://research.character.ai/optimizing-inference/?ref=blog.character.ai\">Character.ai blog</a>." %}

### 구현 예시: JetStream

Google은 [JetStream](https://github.com/google/JetStream)이라는 이 logic을 구현하는 라이브러리를 오픈소스했습니다. 서버는 보통 다른 TPU slice에 있는 "prefill engine" set과 "generate engine" set을 가지며, 이들은 단일 controller에 의해 orchestrate됩니다. Prefill은 "[prefill thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L499)"에서 발생하는 반면, generation은 "[generate thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L629)"에서 발생합니다. 또한 prefill에서 generate slice로 KV cache를 복사하는 것을 orchestrate하는 "[transfer thread](https://github.com/AI-Hypercomputer/JetStream/blob/c0f83127c16d7861cacc560303a28404c6cbb24c/jetstream/core/orchestrator.py#L592)"도 있습니다.

Engine interface([여기](https://github.com/google/JetStream/blob/445f1aa8e857d0a09d72618e365daf80723bdf4c/jetstream/engine/engine_api.py#L138)에서 구현됨)는 모든 LLM이 제공해야 하는 generic interface입니다. 핵심 method는:

* **prefill:** input token set을 가져와서 KV cache를 생성합니다.
* **insert:** KV cache를 가져와서 generate가 생성하고 있는 KV cache batch에 삽입합니다.
* **generate:** batched KV cache set을 가져와서 batch entry당 하나의 token을 생성하고, 각 token에 대해 단일 token의 KV cache를 decode state에 append합니다.

[여기](https://github.com/google/jetstream-pytorch)에서 JetStream의 PyTorch 버전도 사용할 수 있습니다.

## 실습 문제

이 섹션을위해 LLaMA-2 13B를 기반으로 한 새로운 모델을 만들겠습니다. 세부사항은 다음과 같습니다:

| hyperparam        | value  |
| ----------------- | ------ |
| n\_layers (L)     | 64     |
| d\_model (D)      | 4,096  |
| d\_ff (F)         | 16,384 |
| n\_heads (N)      | 32     |
| n\_kv\_heads (K)  | 8      |
| d\_qkv (H)        | 256    |
| n\_embeddings (V) | 32,128 |

**Question 1:** 위 모델은 몇 개의 parameter를 가지고 있나요? Token당 KV cache 크기는 얼마나 됩니까? *Input과 output projection matrix를 공유한다고 가정할 수 있습니다.*

{% details 답을 보려면 클릭하세요. %}

**Parameter count:** 

* MLP parameter count: $$L * D * F * 3$$
* Attention parameter count: $$L * 2 * D * H * (N + K)$$
* Vocabulary parameter: $$D * V$$ (이 matrix들을 공유하므로)

따라서 총 parameter count는 $$L * D * (3F + 2H * (N + K)) + D * V$$입니다. 위의 수를 대입하면 `64 * 4096 * (3*16384 + 2 * 256 * (32 + 8)) + 4096 * 32128 = 18.4e9`입니다. 따라서 이 모델은 약 184억 개의 parameter를 가집니다.

{% enddetails %}

**Question 2:** TPUv5e 4x4 slice에서 이 모델을 서빙하고 이 topology에 걸쳐 KV cache를 완전히 shard할 수 있다고 가정해봅시다. 모든 것에 int8을 사용한다고 가정할 때 맞출 수 있는 가장 큰 batch size는 무엇인가요? KV head 수를 1로 줄인다면 어떨까요?

**Question 3:** 완전히 HBM bandwidth bound라고 가정해봅시다. 모든 parameter를 HBM에서 MXU로 로드하는 데 얼마나 걸리나요? *이는 per-step latency의 좋은 하한입니다.*

**Question 4:** TPUv5e 4x4 slice에서 이 모델을 서빙하고 싶다고 해봅시다. 어떻게 shard할 건가요? *힌트: 먼저 이 질문들에 답해보세요:*

1. ICI에 걸친 이 모델의 tensor parallelism 상한은 무엇인가요?
2. KV cache를 어떻게 shard할 수 있나요?

이 sharding에서 generation의 대략적인 per-step latency는 무엇인가요?

**Question 5:** 위 모델이 실제로 MoE라고 가정해봅시다. MoE 모델은 효과적으로 FFW block의 E copy를 가진 dense model입니다. 각 token은 k개의 FFW block을 통과하고 이 `k`개가 평균화되어 output을 생성합니다. 위 설정에서 `E=16`과 `k=2`를 사용해봅시다.

1. 몇 개의 parameter를 가지고 있나요?
2. FLOP bound가 되는 데 필요한 batch size는 무엇인가요?
3. Token당 KV cache 크기는 얼마인가요(local attention 없다고 가정)?
4. T token으로 forward pass에서 몇 개의 FLOP이 관련되나요?

**Question 6:** MoE에서는 one axis of our mesh에 걸쳐 expert를 split하는 "expert sharding"을 할 수 있습니다. 표준 notation에서 첫 번째 FFW weight는 `[E, D, F]` 형태를 가지고 [E<sub>Z</sub>, D<sub>X</sub>, F<sub>Y</sub>]로 shard하는데, 여기서 `X`는 training 중에만 FSDP dimension으로 사용됩니다. TPU v5e에서 inference를 하고 싶다고 해봅시다:

1. Y=8, Z=16인 TPU v5e 8x16 slice에서 위 모델의 HBM weight loading time은 무엇인가요? TPU당 얼마나 많은 free HBM을 사용할 수 있나요?
2. 모델을 맞출 수 있는 가장 작은 slice는 무엇인가요?

**Question 7 [2D model sharding]:** 여기서는 [ESTI paper](https://arxiv.org/pdf/2211.05102)가 2D weight-stationary sharding이라고 부르는 것의 수학을 다뤄보겠습니다. Appendix B에서 이를 간략히 설명하지만, 수학을 알아낼 수 있는지 보기 위해 이 문제를 먼저 해보세요. 2D weight stationary sharding의 기본 아이디어는 각 chunk가 대략 정사각형이 되도록 $$D$$와 $$F$$ axis 모두를 따라 weight를 shard하는 것입니다. 이는 comm load를 줄이고 약간 더 멀리 확장할 수 있게 해줍니다.

다음은 2D weight stationary algorithm입니다:

<div markdown=1 class="algorithm">

1.  In[B, D<sub>X</sub>] = **AllGather**<sub>YZ</sub>(In[B, D<sub>XYZ</sub>])
2.  Tmp[B, F<sub>YZ</sub>] {U.X} = In[B, D<sub>X</sub>] \*<sub>D</sub> W<sub>in</sub>[D<sub>X</sub>, F<sub>YZ</sub>]
3.  Tmp[B, F<sub>YZ</sub>] = **AllReduce**<sub>X</sub>(Tmp[B, F<sub>YZ</sub>] {U.X})
4.  Out[B, D<sub>X</sub>] {U.YZ} = Tmp[B, F<sub>YZ</sub>] \*<sub>F</sub> W2[F<sub>YZ</sub>, D<sub>X</sub>]
5.  Out[B, D<sub>XYZ</sub>] = **ReduceScatter**<sub>YZ</sub>(Out[B, D<sub>X</sub>] {U.YZ})
</div>

목표는 이 algorithm의 $$T_\text{math}$$와 $$T_\text{comms}$$를 계산하고 언제 전통적인 3D model sharding을 능가할지 찾는 것입니다.

{% details 답을 보려면 클릭하세요! %}

$$T_\text{math}$$와 $$T_\text{comms}$$를 계산해봅시다. 모든 FLOP이 완전히 sharded되므로 이전과 같이 $$T_\text{math} = 4BDF / (N \cdot C)$$를 가지지만 comm은 이제

$$\begin{align*}
T_\text{2D comms} = \frac{2BD}{2X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}} + \frac{2BD}{2X \cdot W_\text{ici}} = \frac{2BD}{X \cdot W_\text{ici}} + \frac{4BF}{YZ \cdot W_\text{ici}}
\end{align*}$$

여기서 AllReduce가 두 배 비싸고 각 operation이 수행되는 axis 수로 comm을 scale한다는 점에 주목합니다. Topology를 선택할 자유가 있고 $$F=4D$$(LLaMA-2에서와 같이)라고 가정하면, (기본 미적분학으로) $$X$$, $$Y$$, $$Z$$의 최적값은 $$X = \sqrt{N / 8}$$, $$YZ = \sqrt{8N}$$이므로 총 communication은

$$T_\text{2D comms} = \frac{2B}{W_\text{ici}} \left(\frac{D}{X} + \frac{8D}{YZ}\right) = \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \approx \frac{11.3 BD}{\sqrt{N} \cdot W_\text{ici}}$$

먼저, 위에서 복사하면, 일반 1D model parallelism은 $$T_\text{model parallel comms} = 4BD / (3 \cdot W_\text{ici})$$를 가질 것이므로, 새로운 comm이 언제 더 작은가요? 우리는

$$\begin{align*}
T_\text{model parallel comms} > T_\text{2D comms} \iff \frac{4BD}{3 \cdot W_\text{ici}} > \frac{\sqrt{128} BD}{\sqrt{N} \cdot W_\text{ici}} \\
\iff N > 128 \cdot \left(\frac{3}{4}\right)^2 = 81
\end{align*}$$

일반적인 $$F$$에 대해서는 이 조건이

$$N > 32 \cdot \left(\frac{F}{D}\right) \cdot \left(\frac{3}{4}\right)^2$$

라고 주장합니다.

그러므로 81개 이상의 chip이 있다면 이 새로운 scheme을 사용하는 것이 더 낫다고 알려줍니다. 이는 역사적으로 약 20 way tensor parallelism에서 ICI bound된다는 것을 발견했기 때문에 약간 이상한 결과입니다. 하지만 여기서는 communication-bound이더라도 총 communication이 총 chip 수와 함께 계속 감소합니다! 이것이 알려주는 것은 chip을 계속 늘리고, batch size를 늘리고, 더 많은 parameter scaling을 하고, 감소된 latency를 볼 수 있다는 것입니다.

{% enddetails %}

<h3 markdown=1 class="next-section">Part 7은 여기까지입니다! TPU에서 LLaMA 3를 어떻게 서빙할지 살펴보는 Part 8을 보려면 [여기](../applied-inference-kr)를 클릭하세요.</h3>

## 부록

### 부록 A: batch size > 240 규칙이 얼마나 실제적인가?

위에서 제공한 간단한 규칙, 즉 compute-bound가 되려면 batch size가 240 token보다 커야 한다는 것은 대략 맞지만 다른 operation이 사용 가능한 모든 HBM을 사용하지 않을 때(inter-device communication을 할 때와 같이) TPU가 weight를 prefetch할 수 있는 능력을 무시합니다.

다음은 d<sub>model</sub> 8192, d<sub>ff</sub> 32768, layer당 2개의 matmul만 있는 작은 Transformer에 대한 layer time(microsecond)의 empirical plot입니다. 이는 [이 Colab notebook](https://colab.sandbox.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)에서 나옵니다. Batch 240 정도까지는 step time이 매우 천천히 증가하다가 그 후 선형적으로 증가하는 것을 볼 수 있습니다.

{% include figure.liquid path="assets/img/batch-scaling-latency.png" class="img-fluid img-small" %}

다음은 token / us에서의 실제 throughput입니다. 이는 주장을 상당히 명확하게 만듭니다. 여기서 layer가 4 way sharded된 약 600M parameter이므로, 최소 약 365us의 latency를 예상할 것입니다.

{% include figure.liquid path="assets/img/batch-scaling-throughput.png" class="img-fluid img-small" %}

적어도 이 모델에서는 실제로 data parallel shard당 BS240까지 throughput이 증가하는 것을 봅니다.

### 부록 B: 2D Weight Stationary sharding

Topology가 커질수록, 더 높은 차원의 mesh(TPU와 같은)에 접근할 수 있다면 두 번째 sharding axis를 도입하여 이를 더 정제하는 것이 가능합니다. 이를 "**2D Weight Stationary**"라고 부르며, [Efficiently Scaling Transformer Inference paper](https://arxiv.org/abs/2211.05102)에서 더 자세히 설명되었습니다.

Megatron에서 hidden $$F$$ dimension만 sharding하고 있기 때문에, 1D sharding으로 chip 수가 커지면 $$E$$($$d_\text{model}$$ dimension)보다 상당히 작아질 수 있습니다. 이는 더 큰 batch size에서 MLP의 첫 번째 layer가 적용된 후 hidden dimension에 걸쳐 collective의 일부를 수행하는 것이 더 경제적일 수 있다는 것을 의미합니다.

{% include figure.liquid path="assets/img/2d-weight-stationary.png" class="img-fluid img-small" %}

이 그림은 다음을 보여줍니다:

1. 1D weight-stationary sharding, 즉 Pure Megatron sharding으로, activation이 AllGather 후 완전히 복제되고, weight가 hidden F dimension에 걸쳐 완전히 sharded됩니다.
2. 2D weight stationary sharding으로, weight가 hidden F와 reduction E dimension 모두에 걸쳐 sharded되고, activation이 E dimension에 걸쳐 sharded됩니다. 첫 번째 layer 전에 (yz) axis에서 AllGather를 수행한 다음 (x) axis에서 ReduceScatter를 수행합니다.

Attention layer의 경우, Megatron style sharding도 더 적은 수의 chip에 대해서는 상대적으로 간단합니다. 하지만 Megatron은 $$n_\text{heads}$$ dimension에서 발생하므로, 가능한 sharding 양에 제한을 둡니다. 2D sharding을 (hidden을 shard하는 대신 $$n_\text{heads}$$ dimension을 shard하는) 수정하면, 더 멀리 확장할 수 있는 능력을 얻습니다.

### 부록 C: Latency bound communication

요약으로, [Section 3](../sharding-kr)에서 각 TPU에서 크기 B인 tensor로 AllGather를 수행하는 데 걸리는 시간을 도출했는데, 이는 full duplex bandwidth WICI와 latency Tmin의 1D ring link에서 X chip에 걸친 것입니다.

$$T_{total} = \max\left(\frac{T_{min} \cdot |X|}{2}, \frac{B}{W_{ICI}}\right)$$

큰 B에 대해서는 시스템에 더 많은 chip을 추가할수록 operation을 수행하는 데 필요한 data movement 양과 사용 가능한 총 bandwidth를 동시에 확장하기 때문에 wall clock이 상대적으로 일정하게 유지됩니다.

{% include figure.liquid path="assets/img/all-gather.gif" class="img-fluid" %}

Latency 최적화된 inference 중 상대적으로 적은 양의 data가 이동되기 때문에, activation에 대한 collective는 종종 latency term에 의해 bound됩니다(특히 작은 batch size에 대해). Latency를 쉽게 시각화할 수 있는데, 완료되기 전에 필요한 hop 수를 세어보면 됩니다.

TPU에서 communication의 tensor size-dependent part가 hop당 1 microsecond 미만이면(hop은 인접한 두 device 간의 communication) collective를 실제로 dispatch하는 고정 overhead에 의해 bottleneck될 수 있습니다. `4.5e10` unidirectional ICI bandwidth로, ICI communication은 다음과 같을 때 latency bound가 됩니다: $$(\text{bytes} / n_\text{shards}) / 4.5e10 < 1e-6$$. 8-way Megatron sharding에 대해서는 `buffer_size < 360kB`일 때입니다. **이것은 inference 중에 실제로 그렇게 작지 않습니다:** int8에서 `BS=16`과 `D=8192`로, activation은 `16*8192=131kB`를 사용하므로 이미 latency bound입니다.

<p markdown=1 class="takeaway">**핵심 내용:** comm이 $$\text{total bytes} < W_{ICI} \times 1e-6$$일 때 latency bound가 됩니다. 예를 들어, $$Y$$에 걸친 model parallelism에서 int8에서 $$Y > BD / 45,000$$일 때 bound가 됩니다.</p>

Compute roofline과 병렬선을 그을 수 있습니다 — 일부 작은 operation(comm에서는 latency, matmul에서는 memory bandwidth)의 고정 cost를 발생시키고 있습니다.

### 부록 D: Speculative Sampling

End to end latency를 *정말로* 신경 쓸 때, speculative sampling<d-cite key="spec1"></d-cite><d-cite key="spec2"></d-cite>이라는 추가 기법을 사용할 수 있습니다. 요약하면, 보통 큰 Transformer에서 token을 하나씩 생성합니다:

{% include figure.liquid path="assets/img/spec-sampling1.png" class="img-fluid" %}

Speculative sampling에서는 더 작고 저렴한 모델을 사용해 token을 생성한 다음 큰 모델로 결과를 확인합니다. 이는 *greedy decoding*으로 이해하기 가장 쉽습니다:

{% include figure.liquid path="assets/img/spec-sampling2.png" class="img-fluid" %}

1. 더 작고 저렴한 모델에서 greedily sample합니다. 이상적으로는 더 큰 모델과 match하도록 training된 모델을 사용하는데, 예를 들어 distillation으로, 하지만 n-gram이나 작은 text corpus의 token matching을 사용하는 것처럼 간단할 수도 있습니다.
2. K개의 token을 생성한 후, 큰 모델을 사용해 지금까지 생성한 모든 token에 대한 next-token logit을 계산합니다.
3. Greedily decoding하고 있으므로, 더 작은 모델에서 생성된 token이 모든 가능한 token 중 가장 높은 probability를 가지는지 확인할 수 있습니다. Token 중 하나가 틀렸다면, 가장 긴 올바른 prefix를 가져와서 첫 번째 틀린 token을 올바른 token으로 바꾼 다음 (1)로 돌아갑니다. 모든 token이 올바르다면, 마지막 올바른 logit을 사용해 (1)로 돌아가기 전에 추가 token을 sample할 수 있습니다.

**왜 이것이 latency win인가요?** 이 scheme은 여전히 token당 큰 모델을 통한 하나의 forward pass에 해당하는 FLOP을 수행해야 하지만, 여러 token을 함께 batch할 수 있기 때문에 이 모든 FLOP을 하나의 forward pass에서 수행할 수 있고 *compute-bound*가 아니라는 사실을 활용해 더 많은 token을 무료로 점수화할 수 있습니다.

수락된 모든 token은 평균적으로 FLOP 측면에서 더 비싸지지만(일부는 거부되고 draft model을 호출해야 하므로), 하드웨어에서 더 많은 FLOP을 짜내고 작은 모델은 저렴하므로 전체적으로는 win합니다. 큰 모델에 의해 모든 것이 확인되었으므로 sampling distribution을 전혀 바꾸지 않습니다(비록 non-greedy에 대해서는 정확한 trajectory가 다를 것이지만).

일반 autoregressive sampling의 경우 token/s는 step time과 같습니다. 여기 Arithmetic Intensity 섹션에 따른 이론적 최소 step time에 여전히 얽매여 있습니다(실제로 Speculative Sampling step time은 보통 일반 autoregressive sampling보다 상당히 느리지만, step당 평균적으로 1개 이상의 token을 얻기 때문에 훨씬 나은 token/s를 얻을 수 있습니다).

{% include figure.liquid path="assets/img/spec-sampling3.png" class="img-fluid" caption="<b>그림:</b> 이 그림은 4B parameter drafter(작은 모델)을 가진 Chinchilla(DeepMind의 70B 모델)에 대한 per-step latency와 speculation success rate를 보여줍니다. XSum(자연어 dataset)의 경우, 이상적인 speculation 양은 약 3-4 token ahead인 반면, HumanEval(코딩 dataset)은 더 예측 가능하며 더 적극적인 speculation에서 win을 봅니다." %}

**Non-greedy decoding에서는 어떻게 작동하나요?** 이는 약간 더 복잡하지만, 본질적으로 logit에서 파생된 $$P_{\text{draft model}}(\text{chosen token})$$과 $$P_{\text{target model}}(\text{chosen token})$$을 가지고 있고, 이 probability의 비율이 어떤 threshold보다 작으면 선택된 token을 확률적으로 거부하는 Metropolis-Hastings에서 영감을 받은 algorithm으로 귀결됩니다.

이 [두](https://arxiv.org/abs/2211.17192) [논문](https://arxiv.org/abs/2302.01318)은 이를 동시에 도출했으며 실제로 어떻게 작동하는지에 대한 좋은 예시를 가지고 있습니다.

<p markdown=1 class="takeaway">**핵심 내용:** Speculative sampling은 더 나은 per token latency를 위해 throughput을 trade하는 또 다른 강력한 lever입니다. 하지만 batch size가 제한된 시나리오(예: 작은 하드웨어 footprint나 큰 KV cache)에서는 win-win이 됩니다.</p>