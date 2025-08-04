---
layout: distill
title: "TPU 프로그램 Profiling 방법"
# permalink: /main/
description: "지금까지 이 시리즈는 완전히 이론적이었습니다: 하드웨어 roofline을 기반으로 한 대략적인 계산들. 그 이해는 멀리 갈 수 있지만 많은 최적화는 실용적인 세부사항으로 귀결됩니다: XLA 컴파일러가 어떻게 작동하는지, 그리고 JAX/Tensorboard Profiler와 같은 프로파일링 도구를 사용하여 실패할 때 무엇을 해야 하는지 파악하는 방법. 여기서 이를 논의합니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 9

previous_section_url: "../applied-inference-kr"
previous_section_name: "Part 8: LLaMA Serving"

next_section_url: ../jax-stuff-kr
next_section_name: "Part 10: JAX"

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
  - name: "TPU Software Stack의 거시적 관점"
  - name: "TensorBoard Profiler: 다용도 TPU Profiler"
  - subsections:
    - name: "Trace Viewer"
    - name: "XLA op을 읽는 방법"
    - name: "Graph Viewer"
    - name: "실제(ish) 예제 profile 살펴보기"
    - name: "Memory Profile"
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

## TPU Software Stack의 거시적 관점

Google은 high level JAX 코드부터 low level Pallas나 HLO까지 TPU를 프로그래밍하기 위한 다양한 API를 노출합니다. 대부분의 프로그래머는 JAX 코드만 작성하는데, 이는 TPU에서 효율적으로 실행되도록 자동으로 컴파일되는 추상적인 NumPy-style linear algebra 프로그램을 작성할 수 있게 해줍니다.

다음은 두 matrix를 곱하는 JAX 프로그램의 간단한 예시입니다:

```py
import jax
import jax.numpy as jnp

def multiply(x, y):
  return jnp.einsum('bf,fd->db', x, y)

y = jax.jit(multiply)(jnp.ones((128, 256)), jnp.ones((256, 16), dtype=jnp.bfloat16))
```

`jax.jit`을 호출함으로써, JAX에게 이 함수를 trace하고 ML computation을 위한 platform-agnostic IR인 [StableHLO](https://openxla.org/stablehlo)라는 lower-level IR을 emit하도록 지시합니다. 이는 차례로 XLA 컴파일러에 의해 HLO로 낮춰집니다. 컴파일러는 JAX profile에서 관찰할 수 있는 HLO를 만드는 fusion, layout, 기타 요소를 결정하기 위해 많은 pass를 실행합니다. 이 HLO는 JAX 코드의 모든 핵심 linear algebra operation(matmul, pointwise op, convolution 등)을 LLVM-style graph view로 나타냅니다. 예를 들어, 다음은 위 프로그램의 HLO 축약 버전입니다<d-footnote>이 HLO를 얻으려면 `jax.jit(f).lower(*args, **kwargs).compile().as_text()`를 실행할 수 있습니다.</d-footnote>:

```c
ENTRY %main.5 (Arg_0.1: f32[128,256], Arg_1.2: bf16[256,16]) -> f32[16,128] {
  %Arg_1.2 = bf16[256,16]{1,0} parameter(1), metadata={op_name="y"}
  %convert.3 = f32[256,16]{1,0} convert(bf16[256,16]{1,0} %Arg_1.2),
  %Arg_0.1 = f32[128,256]{1,0} parameter(0), metadata={op_name="x"}
  ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1},
}
```

HLO의 구문은 곧 설명하겠지만, 지금은 실제로 위의 JAX 코드와 상당히 잘 일치한다는 점만 주목하세요. 예를 들어,

```c
ROOT %dot.4 = f32[16,128]{1,0} dot(f32[256,16]{1,0} %convert.3, f32[128,256]{1,0} %Arg_0.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
```

는 각각 0과 1 dimension을 따라 두 f32 matrix를 곱하는 위의 실제 matmul입니다.

**이 HLO를 TPU에서 실행할 수 있는 코드로 변환하기 위해, XLA 컴파일러는 먼저 이를 LLO**(low-level optimizer) IR로 낮춥니다. LLO는 TPU를 직접 프로그래밍하여 memory 간 copy를 스케줄링하고, systolic array에 배열을 push하는 등의 작업을 수행합니다. LLO 코드는 buffer를 systolic array에 push하고, 결과를 pull하고, TPU memory의 서로 다른 부분 간에 통신하는 DMA를 스케줄링하는 primitive를 포함합니다. 이것이 LLO로 낮춰지면, TPU IMEM에 로드되어 실행되는 machine code로 컴파일됩니다.

프로그램이 원하는 것보다 느리게 실행될 때, 주로 JAX 수준에서 성능을 개선하기 위해 작업합니다. 하지만 그렇게 하려면 종종 HLO의 일부 의미론과 코드가 실제로 TPU에서 어떻게 실행되는지 이해해야 합니다. lower level에서 문제가 발생할 때는 또 다른 escape hatch를 사용하여 [Pallas](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)에서 custom kernel을 작성합니다. 프로그램의 HLO와 runtime 통계를 보기 위해 JAX profiler를 사용합니다.

## TensorBoard Profiler: 다용도 TPU Profiler

JAX는 프로그램이 실행될 때 TPU에서 무슨 일이 일어나는지 이해하기 위한 유용한 도구들이 많이 있는 다용도 TPU profiler를 제공합니다. `jax.profiler` module을 사용하여 실행 중인 프로그램을 trace하고 각 subcomponent의 지속시간, 각 프로그램의 HLO, memory 사용량 등 모든 것을 기록할 수 있습니다. 예를 들어, 이 코드는 TensorBoard에서 볼 수 있는 trace를 `/tmp/tensorboard`의 파일에 dump합니다([여기](https://docs.jax.dev/en/latest/profiling.html#tensorboard-profiling)에 단계별 가이드가 있습니다).

```python
import jax
with jax.profiler.trace("/tmp/tensorboard"):
  key = jax.random.key(0)
  x = jax.random.normal(key, (1024, 1024))
  y = x @ x
  y.block_until_ready()

# 이제 Google Colab에서 TensorBoard를 로드할 수 있습니다
#
# !pip install tensorboard-plugin-profile
# %load_ext tensorboard
# %tensorboard --logdir=/tmp/tensorboard
#
# 또는 외부에서
#
# > tensorboard --logdir=/tmp/tensorboard
#
```

다음은 profiler에서 할 수 있는 것의 개요입니다:

{% include figure.liquid path="assets/img/xprof-overview.png" class="img-fluid" %}

TensorBoard에 들어가면, profiler는 프로그램을 이해하는 데 도움이 되는 몇 가지 핵심 탭을 가지고 있습니다:

1. **Trace Viewer**는 timeline으로 TPU에서 실제로 일어나는 일의 자세한 timeline을 보여줍니다.
2. **Graph Viewer**는 HLO graph를 보여주어 프로그램의 어떤 부분이 서로에게 feeding되는지, 어떻게 sharding되는지 볼 수 있게 해줍니다.
3. **Memory Profile과 Memory Viewer:** 이들은 프로그램이 얼마나 많은 memory를 사용하는지 보여줍니다.

Profile을 공유하기는 약간 어렵지만, [여기](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)에 간단한 Transformer에 대한 적어도 Trace Viewer component를 포함하는 Perfetto 링크가 있습니다. [이 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)을 사용하면 전체 JAX/TensorBoard trace를 생성하고 가지고 놀 수 있습니다.

### Trace Viewer

**Trace Viewer는 아마 profiler의 가장 유용한 부분일 것입니다.** 아래 예시는 annotation이 있는 간단한 Transformer를 보여줍니다. 이름은 코드에 제공된 label에서 나옵니다.

{% include figure.liquid path="assets/img/trace-viewer.png" class="img-fluid" %}

Trace Viewer는 각 TPU core에서 모든 action의 시간순 timeline을 보여줍니다. 일반적으로 모든 TPU가 동일한 instruction을 실행하므로 여기서는 TPU:0만 보고 있습니다. 몇 가지 핵심 참고사항:

1. 맨 위 행(XLA Ops)은 실제 TPU operation(이름은 HLO 이름)을 보여줍니다. 다른 모든 것은 `jax.named_scope`, `jax.named_call`, Python stack trace를 기반으로 한 대략적인 trace입니다.
2. 반복되는 block을 주목하면서, 여기서 단일 layer를 분리할 수 있습니다. 또한 (코드를 보거나/Transformer가 어떻게 작동하는지 이해함으로써) 어떤 부분이 attention이고 어떤 부분이 MLP인지 볼 수 있습니다.
3. XLA op을 클릭하면, 코드에서 어디서 나왔는지 볼 수 있고(trace를 이해하는 데 유용) Graph viewer에 대한 링크를 볼 수 있습니다.

<p markdown=1 class="takeaway">**팁:** A/D로 좌우 panning, W/S로 zoom in/out하는 "비디오 게임" 스타일 컨트롤을 사용하여 Trace Viewer를 탐색할 수 있습니다. 이러한 컨트롤은 탐색을 훨씬 쉽게 만듭니다.</p>

### XLA op을 읽는 방법

HLO는 실제로 읽기가 매우 어렵지 않으며, 위 trace의 주어진 부분이 무엇에 해당하는지 이해하는 데 매우 도움이 됩니다. 다음은 fusion.3이라는 예시 op입니다.

```py
%fusion.3 = bf16[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)} fusion(bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32), kind=kCustom, calls=%all-reduce-scatter.3
```

이를 구성 요소로 분해해봅시다.

* **Op Name**: fusion.3 
  * dot 또는 fusion op은 최대 1개의 matrix multiplication과 아마도 많은 관련 pointwise VPU-op을 포함하는 operation set입니다.
* **Shape/layout**: `bf16[32,32,4096]`
  * 이는 op의 output shape입니다. dtype이 bf16(parameter당 2 byte)이고 `[32,32,4096]`이 shape임을 볼 수 있습니다.
* **Layout:** `{2,1,0:T(8,128)(2,1)}`
  * `{2,1,0:T(8,128)(2,1)}`는 memory에서 axis의 순서(column major, row major 등)와 배열 padding을 알려줍니다. 아래에 더 자세히 설명.
* **Memory location:** S(1) 
  * S(1)은 이 배열이 VMEM에 살고 있음을 알려줍니다. S(0)(때로는 생략됨)은 HBM입니다. S(2)와 S(3)은 다른 memory space입니다.
* **Arguments**: `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)} %fusion.32`
  * 이 op은 특정 shape을 가진 fusion.32라는 bf16 배열인 하나의 입력을 가집니다. 이는 어떤 함수가 이것에 feeding되는지 알려줍니다.

이 notation을 좀 더 이해해봅시다. 간단한 예시로 다음을 들어보겠습니다:

`f32[3,5]{1,0:T(2,2)}`

이는 다시 이 Op이 특정 tiling `{1,0:T(2,2)}`를 가진 shape `[3, 5]`의 float32 배열을 반환한다고 알려줍니다. Tiling은 *너무* 중요하지는 않지만, 간단히 말해서 tiling은 N차원 배열이 memory에서 순차적으로 어떻게 배치되는지 알려줍니다. 다음은 이 배열이 어떻게 배치되는지 보여주는 diagram입니다:

{% include figure.liquid path="assets/img/tiling.png" class="img-fluid" %}

`{1,0:T(2,2)}` 내에서, `1,0` 부분은 physical memory에서 가장 minor부터 가장 major까지 배열 dimension의 순서를 알려줍니다. 이 부분을 오른쪽에서 왼쪽으로 읽고 `f32[3,5]`에서 해당 dimension을 찾아 배열의 physical layout을 파악할 수 있습니다. 이 예시에서 physical layout은 logical shape과 동일한 `[3,5]`입니다.

그 후, `T(2,2)`는 배열이 `(2, 2)` chunk로 tiling되어 있고, 각 chunk 내에서 배열이 rows first(**row-major**), 그 다음 column, 즉 `(0, 0)` 다음에 `(0, 1)`, 그 다음 `(1, 0)`과 `(1,1)`임을 알려줍니다. `T(2, 2)` tiling 때문에 배열은 `[4, 6]`으로 padding되어 memory 사용량이 약 1.6배 확장됩니다. 위에 주어진 큰 bf16 배열 `bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}`의 경우, `T(8,128)(2,1)`을 하는데 이는 배열이 outer `(8, 128)` tiling과 그 unit 내의 inner `(2, 1)` tiling(bf16을 위해 사용되어 load가 항상 4 byte의 배수가 되도록 함) 두 level의 tiling을 가진다고 알려줍니다. 예를 들어, 다음은 `bf16[4,8]{1,0,T(2,4)(2,1)}`입니다(색상은 (2,4) tile, 빨간 상자는 (2,1) tile):

{% include figure.liquid path="assets/img/tiling2.png" class="img-fluid img-small" %}

Tiling은 tensor의 chunk가 VMEM에 얼마나 효율적으로 로드될 수 있는지에 영향을 줄 수 있고, XLA는 때로는 프로그램 내에서 tensor를 "retile"하거나 "re-layout"하는 copy를 도입하기도 하며, 때로는 trivial하지 않은 overhead가 있습니다.<d-footnote>JAX는 XLA가 프로그램의 입력에 대한 "선호하는" layout을 계산할 수 있도록 하는 실험적 기능을 제공하여 이 문제를 해결합니다. `jax.jit`으로 프로그램을 "just-in-time" 컴파일할 때, 일반적으로 JAX에게 어떤 shape과 dtype을 예상하는지 알려주는 "mock" 입력을 전달합니다. 이들은 일반적으로 최적이 아닐 수 있는 tiling 정보도 가지고 있습니다. 대신, 입력 layout을 AUTO로 지정할 수 있고, `jax.jit`은 jitted 프로그램이 선호하는 layout을 반환할 것입니다. 그러면 프로그램 내에서 copy를 유도하는 것을 피하기 위해 그 layout에서 tensor를 명시적으로 로드할 수 있습니다.</d-footnote>

### Graph Viewer

위의 일부 fusion이 복잡해 보일 수 있지만, XLA Graph Viewer가 이를 더 쉽게 parse할 수 있게 만듭니다. 예를 들어 다음은 상당히 복잡한 fusion의 view입니다:

{% include figure.liquid path="assets/img/graph-viewer.png" class="img-fluid" %}

많은 HLO graph를 응시하고 HLO op을 profiling하고 있는 코드에 mapping하려고 시도하는 것이 정말 도움이 됩니다. 상자 위로 hover하면 함수가 정의된 코드 줄을 종종 볼 수 있습니다.

### 실제(ish) 예제 profile 살펴보기

[이 Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)에는 가짜 Transformer에 대한 예제 profile이 있습니다. 급하다면 적어도 Trace Viewer를 보기 위한 [Perfetto 링크](https://ui.perfetto.dev/#!/?s=fa9f13b487bde622707c1a503f9227c34594760a)가 여기 있습니다. 무슨 일이 일어나는지 식별할 수 있도록 trace에 `jax.named_scope` 호출로 annotation하는 데 평소보다 더 많은 노력을 기울였습니다.

{% include figure.liquid path="assets/img/transformer-xprof.png" class="img-fluid" %}

Profile을 보고 각 부분이 무엇을 하는지 정말로 이해하려고 노력해보세요. FFW block부터 시작하여 조금 분해해봅시다:

{% include figure.liquid path="assets/img/transformer-ffw.png" class="img-fluid" %}

여기서 FFW block으로 zoom했습니다. up-projection Op이 입력 `bf16[8, 1024, 8192]`와 `bf16[8192, 16384]`, 출력 `bf16[32, 1024, 16384]`를 가진 fusion(matmul)임을 볼 수 있습니다. (이 코드를 작성했기 때문에) 이것이 4-way DP, 2-way MP sharded matmul의 local view라는 것을 알고 있으므로, 실제로는 다음을 수행하고 있습니다:

**X:** `bf16[32, 1024, 8192]` \* **W<sub>in</sub>**: `bf16[8192, 32768]` -> **Tmp**: `bf16[32, 1024, 32768]`

**이것이 얼마나 걸릴 것으로 예상하나요?** 우선, data parallel shard당 batch size가 `8 * 1024 = 8192`이므로 확실히 compute-bound여야 합니다. 이는 8 TPUv2 core(Google Colab에서 무료로 사용 가능)에서 실행되므로, 약 `2 * 32 * 1024 * 8192 * 32768 / (23e12 * 8) = 95.6ms`가 걸릴 것으로 예상되며, 이는 실제로 걸리는 시간(96ms)과 정확히 일치합니다. 훌륭합니다! 이는 환상적인 FLOP 활용도를 얻고 있다는 것을 의미합니다!

**Communication은 어떨까요?** 두 번째 matmul 끝에 숨겨진 작은 fusion을 알 수 있을 것입니다. 클릭하면 다음을 볼 수 있습니다:

```py
%fusion.1 = bf16[8,1024,4096]{2,1,0:T(8,128)(2,1)} fusion(bf16[8,1024,8192]{2,1,0:T(8,128)(2,1)} %fusion.31), kind=kCustom, calls=%all-reduce-scatter.1
```

이는 기본적으로 작은 ReduceScatter입니다(GraphViewer는 여기):

{% include figure.liquid path="assets/img/reduce-scatter-xprof.png" class="img-fluid" %}

이것이 얼마나 걸릴 것으로 예상하나요? 1.2e11 bidirectional bandwidth에서 한 hop만 필요한 TPUv2 4x2에서 ReduceScatter를 수행하고 있습니다. 배열은 batch axis가 4 way sharded된 크기 `2*32*1024*8192`를 가지므로, 각 shard는 `2*8*1024*8192=134MB`입니다. 따라서 이는 대략 1.1ms가 걸려야 합니다. **실제로는 얼마나 걸릴까요?** Profile에서 보고된 1.13ms입니다. 따라서 roofline에 정말 가깝습니다!

**Attention도 살펴봅시다!** 다음은 attention component의 profile입니다:

{% include figure.liquid path="assets/img/attn-xprof.png" class="img-fluid" %}

Q projection op을 클릭했는데, 이는 shape [d<sub>model</sub> = 8192, n<sub>heads</sub> = 32, d<sub>qkv</sub> = 256]의 matrix $$W_Q$$를 사용합니다. head dimension을 따라 Megatron sharding을 하고 있습니다. 이들이 얼마나 걸려야 하는지 계산하는 동일한 연습을 해보세요.

### Memory Profile

Memory Profile은 시간 함수로서 프로그램 memory를 쉽게 볼 수 있게 만듭니다. 이는 OOM을 debugging하는 데 도움이 됩니다. 여기서 모델 parameter에 할당된 약 7.5GB와 약 10GB free를 볼 수 있습니다. 따라서 memory에 훨씬 더 많은 것을 맞출 수 있습니다.

{% include figure.liquid path="assets/img/memory-viewer.png" class="img-fluid" %}

## 실습 문제

**Question 1**: [이](https://colab.research.google.com/drive/1LfLO3OTr-_MWFPxUN36KJ3cqH0BcAoli?usp=sharing) Colab/profile을 보고 무엇이 의심스러워 보이는지, 여기서 무슨 일이 일어나는지 파악해보세요. 정확히 어떤 computation이 일어나고 있고 각 operation이 무엇을 하는지 말해줄 수 있나요? 관련된 각 matrix의 진짜 shape은 무엇이고 어떻게 sharding되어 있나요? *코드를 읽기 전에 먼저 profile을 보려고 해보세요.*

{% include figure.liquid path="assets/img/all-reduce-profile.png" class="img-fluid" %}

{% details 답을 보려면 클릭하세요. %}

이는 두 개의 matrix multiplication입니다. 구체적으로 다음입니다:

```py
def matmul(w1, w2, x):
  return jnp.einsum('wf,bf->bw', w2, jnp.einsum('fw,bw->bf', w1, x))
```

reduce, 두 개의 큰 fusion, all-reduce를 볼 수 있습니다. 첫 번째 큰 fusion은:

```%fusion.1 = bf16[4096]{0:T(1024)(128)(2,1)} fusion(bf16[4096,8192]{1,0:T(8,128)(2,1)} %param.1, bf16[8192]{0:T(1024)(128)(2,1)} %reduce.6), kind=kLoop, calls=%fused_computation.1```

이는 per-shard shape이 `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`(8192 dimension에 걸쳐)임을 알려줍니다. `replica_groups=\{\{0,16,32,48,64,80,96,112\}, ...\}`를 가진 최종 AllReduce를 관찰함으로써, 8-way model parallelism을 하고 있다는 것을 알 수 있으므로, 진짜 shape은 `[8, 8192] * bf16[32,768, 8192] -> bf16[8, 32,768]`입니다.

{% enddetails %}

**Question 2:** [앞서의 Transformer Colab](https://colab.research.google.com/drive/1_6krERgtolH7hbUIo7ewAMLlbA4fqEF8?usp=sharing)은 간단한 mock Transformer를 구현합니다. Colab의 instruction을 따라 GSPMD partitioning을 가진 naive Transformer의 benchmark를 얻으세요. 각 부분이 얼마나 걸리나요? 얼마나 걸려야 하나요? 어떤 sharding이 사용되고 있나요? Sharding을 고쳐보세요! *힌트: behavior를 제한하기 위해 `jax.lax.with_sharding_constraints`를 사용하세요. 이 수정으로 얻을 수 있는 최고의 MXU는 무엇인가요?*

참고로, 초기 버전은 대략 184ms / layer를 얻고 최적화된 profile은 67 ms / layer를 얻습니다. 이를 완료한 후, profile을 응시하고 purely profile에서 이러한 질문에 답할 수 있는지 보세요:

- 이것은 어떤 sharding 전략인가요?
- Batch size, $$d_\text{model}$$, $$d_\text{ff}$$는 무엇인가요?
- Attention vs. MLP block에 시간의 몇 퍼센트가 소비되나요?
- Roofline에서 각 op에 몇 퍼센트 시간이 소비되어야 하나요?

**참고:** 이 문제가 작성된 이후 XLA 컴파일러가 더 나아졌습니다. 초기 버전은 이제 대략 90ms / layer이고 최적화된 profile은 약 10ms / layer만 더 나을 뿐입니다(80 ms / layer). 여전히 가지고 놀고 더 잘할 수 있는지 보는 것은 가치가 있습니다.

<h3 markdown=1 class="next-section">Part 9는 여기까지입니다. JAX parallelism에 대한 심화 탐구인 Part 10을 보려면 [여기](../jax-stuff-kr)를 클릭하세요.</h3>