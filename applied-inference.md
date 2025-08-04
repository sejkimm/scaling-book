---
layout: distill
title: "TPU에서 LLaMA 3-70B Serving"
# permalink: /main/
description: "TPU v5e에서 LLaMA 3-70B 모델을 어떻게 서빙할지 자세히 살펴보겠습니다. Roofline에서 서로 다른 모델을 서빙하는 비용은 얼마나 될까요? KV cache는 얼마나 클까요? 어떤 batch size를 사용해야 할까요? Inference 중에 parameter와 activation은 어떻게 sharding될까요? Production에서 latency와 throughput에 대한 대략적인 추정을 해보겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 8

previous_section_url: "../inference-kr"
previous_section_name: "Part 7: Inference"

next_section_url: ../profiling-kr
next_section_name: "Part 9: Profiling"

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
  - name: "LLaMA Serving Story는 무엇인가?"
  - subsections:
    - name: "Throughput에 대한 고찰"
    - name: "Prefill은 어떤가?"
  - name: "Latency Throughput Tradeoff 시각화"
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

*이 섹션에서는 LLaMA-3를 서빙하는 데 필요한 조건과 얼마나 효율적으로 수행할 수 있는지 살펴볼 것입니다. 이전 "applied" 섹션에서와 같이, 답을 찾아보기 전에 펜과 종이로 직접 답을 계산해보세요!*

## LLaMA Serving Story는 무엇인가?

LLaMA 3-70B가 어떻게 생겼는지 다시 상기해봅시다([Section 6](../applied-training-kr) 참조):

| **hyperparam**              | **value** |
| --------------------------- | :-------: |
| $$n_\text{layers}$$ (L)     |    80     |
| $$d_\text{model}$$ (D)      |   8,192   |
| $$d_{ff}$$ (F)              |  28,672   |
| $$n_\text{heads}$$ (N)      |    64     |
| $$n_\text{kv heads}$$ (K)   |     8     |
| $$d_\text{qkv}$$ (H)        |    128    |
| $$n_\text{embeddings}$$ (V) |  128,256  |

간단한 질문부터 시작해봅시다: **어떤 하드웨어에서 서빙해야 할까요?** 답은 기본적으로 FLOPs / dollar에서 가장 저렴한 것입니다.<d-footnote>이것이 항상 맞는 것은 아닙니다. 때로는 FLOP보다 더 많은 HBM이나 ICI bandwidth가 중요할 수 있지만, 이는 좋은 heuristic입니다.</d-footnote> 이런 이유로, 일반적으로 현재 전용 inference chip인 TPU v5e에서 서빙하고 싶습니다(비용은 2025년 2월 기준 [Google Cloud pricing](https://cloud.google.com/tpu/pricing)에서):

| **TPU type** | **bfloat16 FLOPs/s** | **Google Cloud USD / hour** | **FLOPs / $** |
| ------------ | :------------------: | :-------------------------: | :-----------: |
| H100         |        9.9e14        |            $10.8            |    3.3e17     |
| v5p          |       4.59e14        |            $4.2             |    3.9e17    |
| v5e          |       1.97e14        |            $1.2             |  **5.8e17**  |

각 TPU v5e는 16GB의 HBM을 가지며, 이는 모델을 상당히 적극적으로 shard해야 합니다. 우리에게 중요할 수 있는 몇 가지 기본적인 양에 대해 생각해보는 것부터 시작해봅시다:

**Question:** LLaMA 3-70B의 token당 KV cache 크기는 얼마나 됩니까? *int8에 저장한다고 가정할 수 있습니다. 이는 주어진 topology에서 batch size가 얼마나 클 수 있는지를 결정합니다.*

{% details 생각해본 후 여기를 클릭하세요! %}

LLaMA 3-70B는 8개의 KV head를 가지므로, token당 크기는 `2 * K * H * L = 2 * 8 * 128 * 80 = 160kB`입니다.

**이것이 얼마나 큰지 주목하세요!** 32k token의 sequence length를 가진다면(일반적임), 이는 `162e3 * 32,768 = 5.3GB / sequence`를 사용합니다. BS=240의 경우, 이는 1.3TB입니다! TPU v5e는 각각 16GB만 가지므로, 이 정도의 memory를 맞추려면 약 `(70e9 + 1.3e12) / 16e9 = 86`개의 TPU v5e chip이 필요할 것입니다. 또한 이것이 70GB의 모델 parameter에 비해 얼마나 큰지도 주목하세요.

{% enddetails %}

**Question:** L3 70B를 batch size 32와 8192 sequence length로 모든 것(param과 KV)을 int8로 서빙하고 싶다고 합시다. 총 memory 사용량은 얼마가 될까요? 이를 서빙할 수 있는 가장 작은 slice는 무엇인가요?

{% details Answer %}

KV가 int8에서 `160e3` byte이므로, 총 KV memory는 `160e3 * 8192 * 32 = 41.9e9` byte입니다. Parameter는 parameter당 1 byte이므로 `70e9` byte입니다. 따라서 총 memory 사용량은 `41.9e9 + 70e9 = 112GB`입니다.

사용할 수 있는 가장 작은 slice는 `112e9 / 16e9 = 7` TPU, 또는 (짝수 크기로 반올림하여) TPU v5e `4x2`가 될 것입니다. 이는 tight fit이고 다른 overhead를 고려하면 완전히 맞지 않을 수 있으므로, 최소한 `4x4`가 필요할 수 있습니다(또는 batch size를 줄여야 함).

{% enddetails %}

**Question:** TPU v5e `4x2`에서 이 batch size와 quantization으로, decode step당 대략 어떤 latency를 예상할 수 있을까요? 어떤 throughput(tokens / sec / chip)을 예상할 수 있을까요? `4x4`는 어떨까요? *bfloat16에서 FLOP을 수행하고 모든 것이 완전히 sharded된다고 가정하세요.*

{% details Answer %}

이전 섹션의 공식을 적용할 수 있습니다:

$$\begin{align*}
\tiny \text{Theoretical Step Time (General)} = \underbrace{\frac{\text{Batch Size} \times \text{KV Cache Size}}{\tiny \text{Total Memory Bandwidth}}}_{\text{Attention (always bandwidth-bound)}} + \underbrace{\max\left(\frac{2 \times \text{Batch Size} \times \text{Parameter Count}}{\text{Total FLOPs/s}}, \frac{\text{Parameter Size}}{\text{Total Memory Bandwidth}}\right)}_{\tiny \text{MLP (can be compute-bound)}}
\end{align*}$$

여기서 parameter가 int8이지만 FLOP이 bfloat16이므로 critical batch size는 약 120이 됩니다. RHS maximum을 수동으로 계산할 수도 있지만, 이는 기본적으로 이미 여러 번 해본 계산입니다. **따라서 matmul과 FLOP 모두에서 memory-bound regime에 충분히 들어가 있습니다.**

Memory bandwidth만 엄격히 보면, step time은 기본적으로 `(KV size + param size) / (8 * HBM bandwidth) = 112e9 / (8 * 8.1e11) = 17ms`입니다. **따라서 이론적으로 step time은 약 17ms입니다.** Throughput은 `32 / .017 = 1882 tokens / sec`, 또는 `1882 / 8 = 235 tokens / sec / chip`가 될 것입니다.

한 가지 주의할 점은 matmul에서 ICI bound가 될 수 있는지 확인하는 것입니다. 여기서 2개 axis를 할당할 수 있으므로, 이론적으로 $Y > 2 * F / 2550 = 2 * 28672 / 2550 = 22$일 때 ICI bound이므로, 괜찮습니다!

`4x4`에서 실행한다면, ICI-wise로 여전히 괜찮을 것이므로 latency는 `17 / 2 = 8.5ms`로 떨어지지만, chip당 throughput은 동일하게 유지됩니다.

{% enddetails %}

### Throughput에 대한 고찰

순전히 throughput에 대해 생각하는 데 약간의 시간을 보내봅시다. Throughput을 최적화할 때는 compute bound가 되기를 원하는데, 이는 모든 TPU MXU capacity를 활용하는 데 가까워진다는 것을 의미합니다. 일반적으로 이는 가능한 한 많은 작업을 수행하기 위해 batch size가 가능한 한 크기를 원한다는 것을 의미합니다.

**Question:** TPU v5e에서 bfloat16 weight와 activation을 사용할 때, matmul에서 compute-bound가 되려면 batch size가 얼마나 커야 할까요? int8 weight로 하지만 bfloat16에서 FLOP을 수행한다면? int8 weight와 int8 FLOP은 어떨까요?

{% details Answer %}

Section 7에서 논의한 바와 같이, $B \ll D, F$인 모든 bfloat16 matmul에 대해:

$$\begin{equation*}
T_\text{math} > T_\text{comms} \leftrightarrow \frac{2BDF}{2DF} \geq \frac{\text{TPU bfloat16 FLOPs/s}}{\text{HBM bandwidth}} = 240
\end{equation*}$$

Weight가 int8일 때, 분모에서 factor 2를 잃으므로 $2BDF / DF = 2B > 240$, 즉 $B > 120$, 이전의 critical batch size의 절반입니다. 이는 우리에게 정말 도움이 됩니다! int8 weight와 int8 FLOP을 수행할 때는 TPU FLOPs/s에 대한 int8 값을 사용해야 하는데, 이는 bfloat16의 1.97e14에서 거의 두 배인 3.94e14로 갑니다. 이는 약 $B > 240$에서 시작했던 곳으로 돌아간다는 것을 의미합니다.

int8 weight와 bfloat16 FLOP의 경우는 꽤 일반적인데, parameter를 무손실로 quantization하는 것이 종종 low-precision arithmetic을 수행하는 것보다 쉽기 때문입니다.

{% enddetails %}

**Question:** 8k context로 bfloat16, int8, int4(KV와 parameter 모두)를 사용하여 LLaMA 3-70B를 서빙할 수 있는 가장 작은 TPU v5e topology는 무엇인가요? *이 문제에서는 KV cache를 무시할 수 있을 정도로 작다고 생각할 수 있습니다.*

{% details Answer %}

이는 쉽습니다! 작은 batch size로 괜찮다면 유일한 제한은 parameter memory를 HBM에 맞추는 것입니다. 즉, 단순히 `ceil(num_params * sizeof(dtype) / HBM per TPU`, 또는 `ceil(70e9 * sizeof(dtype) / 16e9)`를 합리적인 topology(2의 배수)로 반올림한 것입니다:

| dtype | param size | KV size / token (bytes) | min TPU v5es | actual min slice | remaining HBM for KV caches | num KV caches @ 8k |
| :---: | :--------: | :---------------------: | :----------: | :--------------: | :-------------------------: | :----------------: |
| bf16  |   140GB    |          324kB          |     8.75     |  4x4 = 16 chips  |             116             |         43         |
| int8  |    70GB    |          162kB          |     4.38     |  4x2 = 8 chips   |             68              |         52         |
| int4  |    45GB    |          81kB           |     2.81     |  2x2 = 4 chips   |             19              |         67         |

꽤 멋집니다! 이는 원한다면 TPU v5e 2x2에서 LLaMA 70B를 맞출 수 있다는 것을 알려줍니다. 단, KV cache 수가 매우 작다는 것을 알 수 있습니다. 그것이 우리의 batch size입니다! 이는 끔찍한 FLOP 활용도를 얻을 것이라는 것을 의미합니다. batch size를 240까지 밀어올리기 위해 더 큰 topology를 사용하면 매우 기쁠 것입니다.

{% enddetails %}

**Question:** 이러한 topology에 맞는 가장 큰 batch size를 사용한다고 가정할 때, 각 generate step에 대해 어떤 latency를 예상할 수 있을까요?

{% details Answer %}

HBM을 모두 채우도록 batch size를 선택하고 있으므로 이것도 쉽습니다! 이는 전체 TPU v5e worth의 byte를 MXU로 로드하는 데 걸리는 시간에 대한 질문일 뿐입니다. 이는 단순히 `v5e HBM / v5e HBM memory bandwidth = 16GB / 8.2e11 = 19ms`이므로, **19ms / step**입니다. Generation의 median length가 512 token이라고 가정하면, 각 decode에 대해 약 9s입니다. 더 작은 batch size로 약간 더 나은 latency를 얻을 수 있다는 점에 주목하세요. 예를 들어 int4의 모델 parameter만 본다면 HBM이 더 이상 가득 차지 않으므로 최소 latency는 약 10ms / step입니다.

{% enddetails %}

<p markdown=1 class="takeaway">**핵심 내용**: 모델의 모든 parameter를 HBM에서 MXU로 로드하는 데 걸리는 시간을 물어봄으로써 항상 decode latency의 하한을 구할 수 있습니다. KV cache가 작을 때는, 각 layer를 단순히 weight를 chunk-by-chunk로 로드한 다음 버리는 것으로 생각할 수 있습니다. 큰 batch size나 많은 inter-device comm을 사용하지 않는 한, 이는 종종 합리적인 bound입니다(1.5x 내에서). Batch size가 더 클 때는 parameter를 지배하므로 KV cache loading도 모델링해야 합니다.</p>

마찬가지로, FLOP-bound regime(예: training이나 big-batch inference)에서는 communication이 없다고 가정하는 $$\text{Total FLOPs} / (N \cdot C) = 2 \cdot \text{param count} \cdot B / (N \cdot C)$$ 하한을 사용할 수 있습니다.

**Question:** 이들 각각에 대해 chip당 어떤 throughput(queries / chip 관점에서)을 제공할까요? *median decode length가 512 token이라고 가정할 수 있습니다.*

{% details Answer %}

이는 cost / token과 정확히 상관관계가 있기 때문에 중요한 질문입니다.

median decode length에 대한 가정을 사용하면, throughput은 단순히 $$B / (\text{per-step latency} \cdot \text{median steps} \cdot N) \approxeq 43 / (0.019 * 512 * N)$$입니다. 이는 대략 $$(4.42 / N)$$ QPS를 제공하므로, $$N$$을 대입하면:

|  dtype   | QPS / chip |
| :------: | :--------: |
| bfloat16 |    0.27    |
|   int8   |    0.66    |
|   int4   |    1.72    |

이는 forward pass의 working memory(activation과 attention에 할당된 memory)를 완전히 무시하므로 다소 낙관적입니다. Flash Attention으로는 말도 안 되는 것은 아니지만, 현실적이지도 않습니다. 실제 수치는 아마 이의 1/2 정도일 것입니다. 절대 최대 throughput을 위해서는 아마 chip 수를 두 배 이상 늘리고 batch size도 상당히 늘리고 싶을 것입니다.

{% enddetails %}

**Question:** 위의 각 예시에 대해 topology를 두 배로 늘린다면 peak throughput이 어떻게 변할까요?

{% details Answer %}

bfloat16에서 4x8 slice를 사용한다면, KV cache를 위해 186GB가 남아 batch size를 161까지 늘릴 수 있을 것입니다. 그러면 step time이 동일하게 유지되므로, `16.54 / num_chips`의 throughput을 가질 것입니다:

|       dtype       | QPS / chip |
| :---------------: | :--------: |
| bfloat16 (on 4x8) |    0.51    |
|   int8 (on 4x4)   |    1.03    |
|   int4 (on 2x4)   |    2.06    |

더 많은 증가는 더 큰 win을 줄 것입니다! 큰 핵심 내용은 **KV cache size로 제한되는 경우, 가장 작은 topology가 모든 경우에서 가장 성능이 좋은 topology는 아니라는 것**입니다.

{% enddetails %}

**Question:** 이제 sharding 문제를 파헤쳐봅시다. bfloat16로 TPU v5e 4x8에서 서빙하고 싶다고 해봅시다. Generation 중 TPU v5e 4x8에서 모델에 어떤 sharding을 사용할까요? Communication bound가 되는 것을 피할 수 있을까요?

{% details Answer %}

이전 섹션에서 논의한 바와 같이, generation 중에는 실제로 하나의 옵션만 있습니다: model parallelism. Communication bound가 되기 전에 얼마나 많이 할 수 있을까요? 이전 섹션에서 논의한 바와 같이, 모델은 대략 다음과 같을 때 communication bound가 됩니다:

$$Y > \frac{F \cdot n_\text{axes}}{2550}$$

LLaMA 3-70B의 경우 `F = 28,672`이므로, 2 axis의 model sharding을 수행한다면 대략 $$Y = 28672 \cdot 2 / 2550 = 22$$가 되므로, 일반적으로 communication bound가 되지 않고 약 16 chip까지 확장할 수 있어서 `4x4`는 사용할 수 있지만 `4x8`은 사용할 수 없습니다. 일반적으로 computation을 완벽하게 overlap하지 않기 때문에, 이 추정치조차 지나치게 낙관적입니다.

**핵심 내용: 실제로는 pure model parallelism으로 4x8에서 서빙할 수 없습니다.** 여기서 할 수 있는 최선은 4x2 또는 _아마도_ 4x4입니다.

하지만 논의한 바와 같이, batch size가 작을 때는 모델이 memory-bandwidth-bound이고 FLOP bound가 아니므로 throughput을 크게 해치지 않으면서 더 많은 model parallelism을 할 수 있는 경우가 많습니다. 이전에 이 값이 대략 $Y=F / (8\cdot B)$라고 했으므로, batch size 64를 수행한다면 이론적으로 ICI-bound가 되기 전에 `Y = 28,672 / (8 * 64) = 56` way model parallelism까지 갈 수 있습니다. 이를 sanity check하기 위해, 단일 matmul에 대한 $T_\text{ici comms}$, $T_\text{hbm comms}$, $T_\text{math}$를 볼 수 있습니다. 명확히:

$$\begin{align*}T_\text{ici comms} = \frac{2BD}{W_\text{ici}} && T_\text{hbm comms} = \frac{2DF}{Y \cdot W_\text{hbm}} && T_\text{math} = \frac{2BDF}{Y \cdot C}\end{align*}$$

`4x8`의 경우, $T_\text{ici comms}$ = `(2 * 64 * 8192) / 9e10 = 11us`, $T_\text{hbm comms}$ = `(2 * 8192 * 28,672) / (32 * 8.1e11) = 18us`, $T_\text{math}$ = `(2 * 64 * 8192 * 28,672) / (32 * 1.97e14) = 4us`가 되므로, 이론적으로 여전히 HBM bandwidth bound이므로 훌륭합니다! *`4x4`에서 `4x8`로 확장하는 것은 throughput 관점에서 도움이 되지 않을 수 있지만, latency는 줄일 것입니다!

int8과 int4 config를 보면, pure model parallelism으로 _할 수 있습니다_. 따라서 quantization이 더 빠른 FLOP을 넘어서 의미 있는 장점을 실제로 제공하는 지점에 도달했습니다: comm-bound가 되기 전에 더 큰 batch size를 사용할 수 있게 해줍니다. **따라서 이 이야기의 끝은 4x8에서 peak throughput을 달성할 수 없지만, int8과 int4 config에 대해서는 pure model parallelism을 할 수 있다는 것입니다*.

{% enddetails %}

<p markdown=1 class="takeaway">**팁**: 유용한 model parallelism의 최대량은 $$d_{ff}$$와 모델을 sharding하는 axis 수에 달려 있습니다. 최대값은 보통 모델 크기에 따라 8과 32 사이입니다. 일부 throughput cost로 latency를 개선하기 위해 이 한계를 넘어 확장할 수 있습니다.</p>

### Prefill은 어떤가?

여기서는 prefill을 대부분 무시했는데, 훨씬 간단하기 때문입니다. 몇 가지 개념을 종합하여 end-to-end 그림에 대해 생각해봅시다.

**Question:** Prefill 중에 40% FLOP 활용도를 달성한다고 가정합시다. 16 TPU v5e chip에서 길이 8192의 prefill이 얼마나 걸릴까요?

{% details Answer %}

8k token에서는 확실히 compute bound이므로 FLOP에 대해서만 추론하면 됩니다. 모델이 `70e9` parameter를 가진다는 것을 알고 있으므로 각 forward pass는 `2 * 70e9 * B` FLOP을 사용합니다. 40% MFU(FLOP 활용도)를 가정하면, 이는 대략 `2 * 70e9 * 8192 / (16 * 1.97e14 * 0.4) = 0.91s`의 runtime을 제공합니다. 이전에 보던 수치와 비교하면, 실제로 꽤 많습니다!

{% enddetails %}

**Question:** median prefill length가 8192 token이고 median decode length가 4096 token이라고 가정합시다. generate batch size가 32라고 하겠습니다. 평균적으로 step당 몇 개의 sequence가 decoding을 완료할까요? 평균적으로 step당 KV cache에서 몇 개의 token이 evict될까요?

{% details Answer %}

이는 꽤 직관적입니다. median decode length가 4096 token이므로, sequence는 대략 1 / 4096 token마다 완료될 것입니다. batch size 32가 주어지면, 이는 step당 `32 / 4096`개의 sequence가 evict된다는 것을 의미합니다. KV cache length가 대략 `8192 + 4096`이므로, 이는 step당 `32 * (8192 + 4096) / 4096 = 96`개의 token이 evict됩니다. 일반적인 공식은 $B * (P + G) / G$이며, 여기서 $P$와 $G$는 prefill과 generate length입니다.

{% enddetails %}

**Question:** median prefill length 8192와 median decode length 512로 disaggregated serving을 한다고 가정합시다. 위에서 bfloat16으로 계산된 prefill과 generate latency를 가정합시다. 둘 다를 완전히 포화시키는 데 필요한 prefill:generate 서버의 비율은 무엇인가요?

{% details Answer %}

이는 꽤 재미있는 질문입니다. $P$를 prefill 서버 수, $G$를 generate 서버 수라고 하겠습니다. 일반적으로 말하면, 이는 `P / prefill_latency` 비율로 sequence를 입력하고 `B * G / (generate_latency * median_decode_length)` 비율로 소비하는 pipeline 문제입니다. prefill step당 `910ms`, batch size 43에서 decode step당 `19ms`를 계산했습니다(32라고 하겠습니다). 따라서 `P / 0.91 = 32 * G / (0.019 * 512)` 또는 `P = 3G`, 즉 generation 서버보다 약 3배 많은 prefill 서버가 필요합니다!

{% enddetails %}

## Latency Throughput Tradeoff 시각화

LLaMA 70B를 잠시 고수하면서, generation 중 다른 batch size에 대한 latency와 throughput을 실제로 살펴봅시다. PaLM model에 대해 이전 섹션에서 보여준 바와 같이, 이는 throughput/latency에 대한 Pareto frontier를 제공합니다. MLP block에서 compute-bound를 유지하면서 사용할 수 있는 합리적인 bound이므로 16-way tensor parallelism을 가정하겠습니다. 여기서는 TPU v5e 4x4 topology를 사용하겠습니다. **슬라이더는 sequence length를 제어하므로 더 큰 KV cache의 효과를 볼 수 있습니다.**

<div class="l-page">
  <iframe src="{{ 'assets/plotly/pareto.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

* **Cost와 latency 간의 tradeoff가 얼마나 극적인지 보세요.** Per-token latency를 두 배로 하는 비용으로, per-token cost에서 대략 100배 감소를 달성할 수 있습니다. 또한 latency는 낮은 batch size에서 5.5ms부터 매우 큰 batch에서 20ms까지 어디서든 범위를 가질 수 있습니다.
* 2k context에서 throughput이 BS 120 roofline에 도달할 때(여기서 120은 int8 weight를 하지만 bf16 FLOP을 하기 때문) 약 1 token / ms / chip에서 효과적으로 plateau한다는 점에 주목하세요. 하지만 sequence length가 증가하면 더 이상 이 batch size를 memory에 맞출 수 없으므로 full saturation 지점에 절대 도달하지 않습니다.
* KV loading이 (parameter loading 대신) 지배적이 되므로 동일한 throughput에 대해 큰 batch size에서 latency가 얼마나 높은지 주목하세요.

Param loading time, KV loading time, FLOP time으로 cost와 latency의 source를 분해하여 이를 더 잘 이해할 수 있습니다. 빨간색 sector는 MLP block에서 compute-bound가 될 것으로 예상되는 영역입니다.

<div class="l-page">
  <iframe src="{{ 'assets/plotly/latency_breakdown_log.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>
</div>

이는 꽤 이야기를 전해줍니다. 처음에는 parameter loading이 latency의 대부분을 차지하다가, batch size가 FLOP과 KV loading이 더 중요해질 만큼 커질 때까지 그렇다는 것을 볼 수 있습니다. 특히, 2048보다 큰 모든 sequence length에서는 FLOP보다 KV cache loading에 더 많은 시간을 소비합니다! **따라서 batch size를 늘려 하드웨어 활용도를 개선할 수 있지만, 긴 context length에서는 KV loading이 항상 총 step time을 지배합니다.**

<p markdown=1 class="takeaway">**핵심 내용:** LLaMA 3-70B의 경우, 이러한 구성의 거의 모든 곳에서 강하게 KV cache memory bandwidth-bound(그리고 HBM-bound)이며, 이는 generation throughput을 위해 KV cache size를 줄이는 것이 얼마나 중요한지를 강조합니다. 또한 여기서 latency/throughput tradeoff가 얼마나 극적으로 유지되는지도 주목하세요.</p>

{% details 이를 위한 코드는 꽤 간단합니다. %}

다음은 이러한 roofline을 계산하는 코드입니다:

```py
import numpy as np

num_chips = 16  # we fix 16 as the amount of total model parallelism we do
param_size = 70e9  # int8 means 1 byte per param
sequence_length = 8192  # can vary this

hbm_bandwidth = 8.20E+11  # v5e
flops = 1.97E+14  # v5e

param_size = bytes_per_param * param_count

def kv_cache_size(bs):
    return 2 * bs * 128 * 8 * 80
    
def min_topology(bytes):
    return 2 ** np.ceil(np.log2(bytes / 16e9))

def get_max_batch_size(max_num_chips: int = 16):
  # for num_chips in topo_sizes:
  batch_sizes = np.arange(1, 1024, 4)
  kv_sizes = kv_cache_size(sequence_length * batch_sizes)
  num_chips = min_topology(kv_sizes + param_size)
  max_idx = np.where(num_chips <= max_num_chips)[0][-1]
  return max_idx

max_idx = get_max_batch_size(num_chips, sequence_length, param_size)  # get the largest batch size that can fit
batch_sizes = np.arange(1, 512, 1)[:max_idx]
kv_sizes = kv_cache_size(sequence_length * batch_sizes)

kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

param_comms_time = param_size / (num_chips * hbm_bandwidth)
param_comms_time = np.asarray([param_comms_time] * batch_sizes.shape[0])

flops_time = 2 * param_count * batch_sizes / (num_chips * flops)  # roughly true in a 2ND sense

mlp_time = np.maximum(flops_time, param_comms_time)
attn_time = kv_comms_time  # always bandwidth-bound for generate

latency = 1000 * (mlp_time + attn_time)
throughput = batch_sizes / (latency * num_chips)
```

latency를 두 source로 매우 명시적으로 분해하는 방법: KV loading과 param loading, 그리고 latency가 FLOP 또는 comm 중 더 큰 것에 의해 bound되는 방법에 주목하세요.

{% enddetails %}

## 실습 문제

다음은 몇 가지 실습 문제입니다. 이 중 일부는 위에서 해결된 것을 반복하지만, 교육적으로 유용할 수 있습니다.

**Question 1:** LLaMA 3-405B의 각 forward pass는 token당 몇 개의 FLOP을 사용하나요? FLOP bound라고 가정할 때, TPU v5e의 N chip에서 단일 forward pass의 하한은 무엇인가요? Comm bound라면? *모델이 단일 chip에 맞지 않는다는 사실을 무시하세요.*

**Question 2:** int8 weight와 int8 KV cache를 사용하여 BS240으로 LLaMA 3-8B를 서빙하고 싶다고 가정합시다. (a) 모델 parameter (b) KV cache (c) peak working activation(대략)에 의해 몇 byte가 사용되나요? 이를 실행할 수 있는 가장 작은 topology는 무엇인가요?

**Question 3:** TPU v5e에서 LLaMA 3-405B를 어떻게 서빙하시겠습니까? int8 weight와 bfloat16 FLOP을 가정하세요. 15ms / token의 확고한 제한이 있다고 하면, 달성할 수 있는 가장 높은 throughput configuration은 무엇인가요? 이론적 최소 step time은 무엇인가요?

<h3 markdown=1 class="next-section">Part 8은 여기까지입니다! XLA와 TPU profiling에 대한 심화 탐구인 Part 9를 보려면 [여기](../profiling-kr)를 클릭하세요.</h3>