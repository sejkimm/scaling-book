---
layout: distill
title: "샤딩된 행렬과 곱셈 방법"
# permalink: /main/
description: "여기서는 가장 큰 ML 모델들이 어떻게 여러 accelerator에 걸쳐 분할(또는 '샤딩')되는지 설명합니다. LLM은 대부분 행렬 곱셈으로 구성되어 있으므로, 이를 이해하는 것은 행렬이 디바이스에 걸쳐 분할되었을 때 어떻게 곱하는지 이해하는 것으로 귀결됩니다. 우리는 TPU 통신 primitive의 비용을 기반으로 한 샤딩된 행렬 곱셈의 간단한 이론을 개발합니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 3

previous_section_url: "../tpus-kr"
previous_section_name: "Part 2: TPUs"

next_section_url: ../transformers-kr
next_section_name: "Part 4: Transformer Math"

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
  - name: "분할 표기법과 Collective Operations"
  - subsections:
    - name: "샤딩을 위한 통합 표기법"
    - name: "잠깐: 이를 코드로 어떻게 설명할까요?"
  - name: "샤딩된 배열을 이용한 계산"
  - subsections:
    - name: "Case 1: 어떤 피승수도 샤딩된 contracting dimension을 가지지 않는 경우"
    - name: "Case 2: 한 피승수가 샤딩된 contracting dimension을 가지는 경우"
    - name: "Case 3: 두 피승수 모두 샤딩된 contracting dimension을 가지는 경우"
    - name: "Case 4: 두 피승수 모두 같은 축을 따라 샤딩된 non-contracting dimension을 가지는 경우"
  - name: "TPU 통신 Primitives에 대한 심층 탐구"
  - subsections:
    - name: "마지막 통신 primitive: AllToAll"
    - name: "ReduceScatter에 대한 자세한 내용"
  - name: "우리가 배운 것"
  - name: "연습할 문제들"

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

## 분할 표기법과 Collective Operations

만 개의 TPU에서 LLM을 훈련할 때, 우리는 여전히 한 개에서 훈련할 때와 추상적으로 같은 계산을 하고 있습니다. 차이점은 **우리의 배열이 단일 TPU의 HBM에 맞지 않기** 때문에 분할해야 한다는 것입니다.<d-footnote>속도를 위해 병렬화를 선택할 수도 있다는 점을 주목할 가치가 있습니다. 더 적은 수의 칩에 맞출 수 있더라도, 더 많은 수로 스케일링하면 단순히 더 많은 FLOP/s를 얻습니다. 예를 들어, 추론 중에는 때때로 더 작은 토폴로지에 맞출 수 있지만 지연 시간을 줄이기 위해 더 큰 토폴로지로 스케일링을 선택합니다. 마찬가지로, 훈련 중에는 스텝 시간을 줄이기 위해 종종 더 많은 칩으로 스케일링합니다.</d-footnote> 우리는 이것을 배열의 "*sharding*" 또는 "*partitioning*"이라고 부릅니다.

다음은 4개의 TPU에 걸쳐 샤딩된 2D 배열 **A**의 예입니다:

{% include figure.liquid path="assets/img/sharding-example.png" class="img-fluid" caption="<b>그림:</b> 모양 <b>A</b>[I, J]의 예제 배열이 4개 디바이스에 걸쳐 샤딩됩니다. 두 차원 모두 샤딩 <b>A</b>[I<sub>X</sub>, J<sub>Y</sub>]로 2개 디바이스에 걸쳐 균등하게 샤딩됩니다. 각 TPU는 총 메모리의 1/4을 보유합니다." %}

샤딩된 배열이 여전히 샤딩되지 않은 배열과 같은 *global* 또는 *logical shape*을 가진다는 점에 주목하세요. 예를 들어 `(4, 128)`이지만, `(2, 64)`와 같은 *device local shape*도 가지는데, 이는 각 TPU가 보유하고 있는 실제 바이트 크기를 알려줍니다 (위 그림에서 각 TPU는 전체 배열의 ¼을 보유). 이제 이를 임의의 배열로 일반화하겠습니다.

### 샤딩을 위한 통합 표기법

우리는 *named-axis notation*의 변형을 사용하여 tensor가 디바이스에 걸쳐 블록으로 *어떻게* 샤딩되는지 설명합니다: 각 축에 **mesh axis names** **예: X**, **Y, Z**가 주어진 **device mesh**라고 하는 2D 또는 3D 디바이스 격자의 존재를 가정합니다. 그러면 배열의 각 명명된 차원이 물리적 mesh 축에 걸쳐 어떻게 분할되는지 설명하여 행렬 데이터가 device mesh에 걸쳐 어떻게 배치되는지 지정할 수 있습니다. 우리는 이 할당을 **sharding**이라고 부릅니다.

**예제 (위 다이어그램)**: 위 다이어그램의 경우:
* **Sharding:** $A[I_X, J_Y]$, 이는 첫 번째 축 $I$를 mesh 축 $X$를 따라, 두 번째 축 $J$를 mesh 축 $Y$를 따라 샤딩하라고 알려줍니다. 이 샤딩은 각 샤드가 배열의 $1 / (\lvert X\rvert \cdot \lvert Y\rvert)$을 보유한다고 알려줍니다.
* **Mesh:** 위의 device mesh `Mesh(devices=((0, 1), (2, 3)), axis_names=('X', 'Y'))`는 축 이름 $X$와 $Y$를 가진 2x2 격자에 4개의 TPU가 있다고 알려줍니다.

종합하면, 배열의 local shape(개별 디바이스가 보유하는 샤드의 크기)가 $(\lvert I\rvert / 2, \lvert J\rvert / 2)$라는 것을 알 수 있습니다. 여기서 $$\lvert I\rvert$$는 A의 첫 번째 차원의 크기이고 $$\lvert J\rvert$$는 A의 두 번째 차원의 크기입니다.

**예제 (1개 축에 걸친 2D 샤딩)**: $A[I_{XY}, J]$는 첫 번째 차원(I)을 X와 Y 하드웨어 축 모두에 걸쳐 샤딩합니다. 디바이스당 바이트 수는 이전 샤딩과 같지만 local shape는 다릅니다. 이제 $(\lvert I\rvert /(\lvert X\rvert \cdot \lvert Y\rvert), \lvert J\rvert)$입니다.

**이러한 샤딩들 시각화하기:** 4개 디바이스에 걸쳐 분할된 2D 데이터 배열을 보면서 이러한 샤딩들을 시각화해봅시다:

{% include figure.liquid path="assets/img/sharding-colored1.png" class="img-fluid img-small" %}

우리는 행렬의 *완전 복제된* 형태를 샤딩 할당 없이 단순히 $A[I, J]$로 씁니다. 이는 *각* 디바이스가 전체 행렬의 완전한 복사본을 포함한다는 의미입니다.

{% include figure.liquid path="assets/img/sharding-colored2.png" class="img-fluid img-small" %}

이러한 차원 중 하나가 mesh 축에 걸쳐 분할되었음을 나타내고 싶을 때, mesh-axis 하첨자를 사용하여 표시합니다. 예를 들어 $A[I_X, J]$는 **I** logical 축이 **X** mesh 차원에 걸쳐 분할되었지만, **J** 차원은 분할되지 *않았고*, 블록들이 **Y** mesh 축에 걸쳐 *부분적으로 복제되어* 남아있다는 의미입니다.

{% include figure.liquid path="assets/img/sharding-colored3.png" class="img-fluid img-small" %}

$A[I_X, J_Y]$는 **I** logical 축이 **X** mesh 축에 걸쳐 분할되고, **J** 차원이 **Y** mesh 축에 걸쳐 분할되었다는 의미입니다.

{% include figure.liquid path="assets/img/sharding-colored4.png" class="img-fluid img-small" %}

아래 그림에서 다른 가능성들을 보여줍니다:

{% include figure.liquid path="assets/img/sharding-colored5.png" class="img-fluid" %}

여기서 $A[I_{XY}, J]$는 **X**와 **Y** mesh 축을 더 큰 평면화된 차원으로 취급하고 **I** 명명된 축을 모든 디바이스에 걸쳐 분할한다는 의미입니다. 여러 mesh-axis 하첨자의 순서는 격자에 걸친 분할의 순회 순서를 지정하므로 중요합니다.

{% include figure.liquid path="assets/img/sharding-colored6.png" class="img-fluid img-small" %}

마지막으로, 여러 명명된 축이 *같은* mesh 차원을 따라 샤딩될 *수 없다*는 점에 주목하세요. 예를 들어 $A[I_X, J_X]$는 말이 안 되는, 금지된 샤딩입니다. mesh 차원이 배열의 한 차원을 샤딩하는 데 사용되면, 어떤 의미에서 "소모된" 것입니다.

<b markdown=1 style="color: #57cf57;">퀴즈:</b> **A**가 모양 `int8[128, 2048]`, 샤딩 $A[I_{XY}, J]$, mesh `Mesh({'X': 2, 'Y': 8, 'Z': 2})` (총 32개 디바이스)인 배열이라고 하겠습니다. **A**는 디바이스당 얼마나 많은 메모리를 사용하나요? 모든 디바이스에 걸쳐 **A**는 총 얼마나 많은 메모리를 사용하나요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 우리 배열 **A**는 X와 Y에 걸쳐 샤딩되고 Z에 걸쳐 복제되므로, 디바이스당 모양이 `int8[128 / (2 * 8), 2048] = int8[8, 2048]`이고, 크기는 `8 * 2048 = 16,384` 바이트입니다. Z에 걸쳐 복제되므로, Z-plane 내에서는 X와 Y에 걸쳐 완전히 샤딩되어 있지만, Z-plane당 하나의 복사본이 있고 2개의 그런 plane이 있으므로, 총 크기(모든 디바이스에 걸쳐)는 `128 * 2048 * 2 = 512kiB`입니다.

{% enddetails %}

### 잠깐: 이를 코드로 어떻게 설명할까요?

JAX는 위에서 설명하는 추상 구문과 매우 유사한 명명된 샤딩 구문을 사용합니다. [섹션 10](../jax-stuff-kr)에서 이에 대해 더 자세히 이야기하겠지만, 여기 간단한 미리보기가 있습니다. [여기](https://colab.research.google.com/drive/15cxw66eABwZPG-V4QFmbLfiykPFf_gaP?usp=sharing) Google Colab에서 이를 실험해보고 결과를 프로파일링하여 JAX가 다른 샤딩을 어떻게 처리하는지 볼 수 있습니다. 이 스니펫은 3가지 일을 합니다:

1. 8개의 TPU를 'X'와 'Y'가 두 축에 할당된 4x2 격자로 매핑하는 **jax.Mesh**를 생성합니다.
2. A가 두 차원 모두에 걸쳐 샤딩되고 B가 출력 차원에 걸쳐 샤딩되는 행렬 A와 B를 생성합니다.
3. 샤딩된 배열을 반환하는 간단한 행렬 곱셈을 컴파일하고 수행합니다.

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

# 우리의 mesh를 생성합니다! 'X'와 'Y'라는 이름을 가진 TPU v2-8 4x2 slice에서 실행하고 있습니다.
assert len(jax.devices()) == 8
mesh = jax.make_mesh(axis_shapes=(4, 2), axis_names=('X', 'Y'))

# 샤딩을 정의하는 데 도움이 되는 작은 유틸리티 함수입니다. PartitionSpec은 우리의
# 샤딩(축에서 이름으로의 매핑)입니다.
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

# A와 B 모두를 non-contracting 차원에 걸쳐 샤딩하고 A를 contracting dim에 걸쳐 샤딩합니다.
A = jnp.zeros((8, 2048), dtype=jnp.bfloat16, device=P('X', 'Y'))
B = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, device=P(None, 'Y'))

# 이러한 샤딩된 배열에서 matmul을 수행할 수 있습니다! out_shardings는 출력이 어떻게
# 샤딩되기를 원하는지 알려줍니다. JAX/XLA가 나머지 샤딩을 처리합니다.
compiled = jax.jit(lambda A, B: jnp.einsum('BD,DF->BF', A, B), out_shardings=P('X', 'Y')).lower(A, B).compile()
y = compiled(A, B)
```

JAX의 멋진 점은 이러한 배열들이 샤딩되지 않은 것처럼 동작한다는 것입니다! `B.shape`는 global 또는 logical shape (2048, 8192)를 알려줍니다. 실제로 로컬로 어떻게 샤딩되어 있는지 보려면 `B.addressable_shards`를 봐야 합니다. 이러한 배열에서 연산을 수행할 수 있고 JAX는 연산을 수행하기 위해 어떻게 브로드캐스트하거나 reshape할지 알아내려고 시도합니다. 예를 들어, 위 예제에서 **A**의 local shape는 `[2, 1024]`이고 **B**의 경우 `[2048, 4096]`입니다. JAX/XLA는 최종 곱셈을 수행하기 위해 필요에 따라 이러한 배열에 걸쳐 자동으로 통신을 추가할 것입니다.

## 샤딩된 배열을 이용한 계산

여러 디바이스에 걸쳐 분산된 데이터 배열이 있고 이에 대해 수학적 연산을 수행하고 싶다면, 데이터와 계산을 샤딩하는 것과 관련된 오버헤드는 무엇일까요?

분명히, 이는 관련된 계산에 따라 다릅니다.

* *Elementwise* 연산의 경우, 분산 배열에서 연산하는 데 **오버헤드가 없습니다**.
* 많은 디바이스에 있는 요소들에 걸쳐 연산을 수행하고 싶을 때, 일이 복잡해집니다. 다행히, 대부분의 머신러닝에서 거의 모든 계산이 행렬 곱셈 형태로 이루어지며, 이들은 분석하기 상대적으로 간단합니다.

이 섹션의 나머지 부분은 샤딩된 행렬을 곱하는 방법을 다룰 것입니다. 1차 근사에서, 이는 각 청크를 완전히 곱하거나 합할 수 있도록 행렬의 청크를 이동하는 것을 포함합니다. **각 샤딩은 다른 통신을 포함할 것입니다.** 예를 들어, $A[I_X, J] \cdot B[J, K_Y] \to C[I_X, K_Y]$는 *contracting dimension* (J, 실제로 합하는 것)이 샤딩되지 않았기 때문에 어떤 통신 없이도 곱할 수 있습니다. 하지만, 출력이 샤딩되지 않기를 원한다면 (즉, $A[I_X, J] \cdot B[J, K_Y] \to C[I, K]$), $A$ 또는 $C$를 모든 디바이스에 복사해야 합니다. 이 두 선택은 다른 통신 비용을 가지므로, 이 비용을 계산하고 가장 낮은 것을 선택해야 합니다.

{% details 이를 "블록 행렬 곱셈"의 관점에서 생각할 수 있습니다. %}

먼저 "블록 행렬" 또는 행렬의 nested 행렬의 개념을 상기해봅시다:

$$\begin{equation}
\begin{pmatrix}
a_{00} & a_{01} & a_{02} & a_{03} \\
a_{10} & a_{11} & a_{12} & a_{13} \\
a_{20} & a_{21} & a_{22} & a_{23} \\
a_{30} & a_{31} & a_{32} & a_{33}
\end{pmatrix}
=
\left(
\begin{matrix}
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11}
\end{bmatrix} \\
\begin{bmatrix}
a_{20} & a_{21} \\
a_{30} & a_{31}
\end{bmatrix}
\end{matrix}
\begin{matrix}
\begin{bmatrix}
a_{02} & a_{03} \\
a_{12} & a_{13}
\end{bmatrix} \\
\begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
\end{matrix}
\right)
=
\begin{pmatrix}
\mathbf{A_{00}} & \mathbf{A_{01}} \\
\mathbf{A_{10}} & \mathbf{A_{11}}
\end{pmatrix}
\end{equation}$$

행렬 곱셈은 행렬 피승수가 블록으로 쓰여질 때, 곱이 표준 규칙을 따르는 블록 matmul의 관점에서 쓰여질 수 있다는 좋은 속성을 가집니다:

$$\begin{equation}
\begin{pmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{pmatrix}
\cdot
\begin{pmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{pmatrix}
=
\begin{pmatrix}
A_{00}B_{00} + A_{01}B_{10} & A_{00}B_{01} + A_{01}B_{11} \\
A_{10}B_{00} + A_{11}B_{10} & A_{10}B_{01} + A_{11}B_{11}
\end{pmatrix}
\end{equation}$$

이것이 의미하는 바는 분산 행렬 곱셈을 구현하는 것이 이러한 샤딩된 블록들을 네트워크를 통해 이동시키고, 블록에서 *로컬* 행렬 곱셈을 수행하고, 결과를 합하는 것으로 귀결된다는 것입니다. **그러면 질문은 어떤 통신을 추가할지, 그리고 그것이 얼마나 비싼지입니다.**

{% enddetails %}

편리하게도, 우리는 모든 가능한 샤딩을 대략 4가지 경우로 요약할 수 있으며, 각각은 어떤 통신을 추가해야 하는지에 대한 규칙을 가집니다
1. **[Case 1](#case-1-어떤-피승수도-샤딩된-contracting-dimension을-가지지-않는-경우):** 어떤 입력도 contracting dimension을 따라 샤딩되지 않은 경우. _어떤 통신 없이 로컬 샤드를 곱할 수 있습니다._
2. **[Case 2](#case-2-한-피승수가-샤딩된-contracting-dimension을-가지는-경우):** 한 입력이 샤딩된 contracting dimension을 가지는 경우. _일반적으로 contracting dimension을 따라 샤딩된 입력을 "AllGather"합니다._  
3. **[Case 3](#case-3-두-피승수-모두-샤딩된-contracting-dimension을-가지는-경우):** 두 입력 모두 contracting dimension을 따라 샤딩된 경우. _로컬 샤드를 곱한 다음, 결과를 "AllReduce"할 수 있습니다._
4. **[Case 4](#case-4-두-피승수-모두-같은-축을-따라-샤딩된-non-contracting-dimension을-가지는-경우):** 두 입력 모두 같은 축을 따라 샤딩된 non-contracting dimension을 가지는 경우. 먼저 두 입력 중 하나를 AllGather하지 않고는 진행할 수 없습니다.

이들을 단순히 따라야 하는 규칙으로 생각할 수 있지만, 이러한 규칙이 왜 성립하고 얼마나 비싼지 이해하는 것도 가치가 있습니다. 이제 이들 각각을 자세히 살펴보겠습니다.

### Case 1: 어떤 피승수도 샤딩된 contracting dimension을 가지지 않는 경우

**보조정리:** 분할된 tensor를 곱할 때, contracting dimension이 샤딩되거나 두 tensor 모두 같은 축을 따라 샤딩된 non-contracting dimension을 가지지 *않는 한* 계산이 유효하고 출력은 입력의 샤딩을 따릅니다. 예를 들어, 이것은 잘 작동합니다

$$\begin{equation*}
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow \mathbf{C}[I_X, K_Y]
\end{equation*}$$

어떤 통신도 전혀 없이, X와 Y 하드웨어 차원 모두에 걸쳐 샤딩된 tensor를 결과로 합니다. 왜 그런지 생각해보세요. 기본적으로, 계산은 샤딩과 *독립적*입니다. 각 배치 항목이 곱하고 축소할 수 있는 축이 축약되는 일부 로컬 청크를 가지기 때문입니다. 이러한 경우들 중 어느 것이든 잘 작동하고 이 규칙을 따릅니다:

$$\begin{align*}
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I, K] \\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K] \rightarrow &\ \mathbf{C}[I_X, K]\\
\mathbf{A}[I, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I, K_Y]\\
\mathbf{A}[I_X, J] \cdot \mathbf{B}[J, K_Y] \rightarrow &\ \mathbf{C}[I_X, K_Y]
\end{align*}$$

**A**나 **B** 모두 샤딩된 contracting dimension **J**를 가지지 않기 때문에, 우리는 단순히 입력의 로컬 블록 행렬 곱셈을 수행할 수 있고 결과는 원하는 출력 샤딩에 따라 *이미* 샤딩될 것입니다. 두 피승수 모두 같은 축을 따라 샤딩된 non-contracting dimension을 가질 때는 더 이상 참이 아닙니다 (자세한 내용은 [유효하지 않은 샤딩](#case-4-두-피승수-모두-같은-축을-따라-샤딩된-non-contracting-dimension을-가지는-경우) 섹션 참조).

### Case 2: 한 피승수가 샤딩된 contracting dimension을 가지는 경우

contracting **J** 차원에서 샤딩된 **A**를 완전히 복제된 **B**에 대한 분산 행렬 곱셈의 간단한 경우를 고려해봅시다:

$$\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

**A**의 contracting axis에서 전체 데이터가 누락되어 있으므로 로컬 **A**, **B** 블록을 서로 단순히 로컬 행렬 곱셈을 수행할 수 없습니다. 일반적으로, 우리는 먼저 **A**의 샤드를 로컬로 "**AllGather**"한 다음, **B**와 곱합니다:

$$\textbf{AllGather}_X[I, J_X] \rightarrow \mathbf{A}[I, J]$$

$$\mathbf{A}[I, J] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$$

AllGather는 축을 따라 *샤딩을 제거*하고 디바이스에 걸쳐 분산된 샤드를 그 축을 따라 *각* 디바이스에 재조립합니다. 위의 표기법을 사용하면, AllGather는 축 집합에서 하첨자를 제거합니다. 예를 들어

$$\textbf{AllGather}_{XY}(A[I_{XY}, J]) \rightarrow A[I, J]$$

또한 주어진 차원에 대해 모든 하첨자를 제거할 필요는 없습니다. 예를 들어 $$A[I_{XY}, J] \rightarrow A[I_Y, J]$$도 AllGather이며, 단지 단일 축에 대해서만입니다.

또한 *non-contracting* 차원 샤딩을 제거하기 위해 AllGather를 사용하고 싶을 수도 있습니다. 예를 들어 행렬 곱셈:

$$A[I_X, J] \cdot B[J, K] \rightarrow C[I, K]$$

마찬가지로 **X**를 따라 AllGather하여 출력 샤딩을 제거할 수 있지만, 이 경우 contracting dimension을 AllGather하는 경우와 달리 행렬 곱셈 전후에 수행할 자유가 있습니다. contracting dimension을 AllGather하는 경우에는 행렬 곱셈을 수행하기 전에 강제로 수행해야 합니다.

**AllGather는 실제로 어떻게 수행될까요?** 단일 축을 따라 AllGather를 수행하려면, 모든 디바이스가 복사본을 가질 때까지 축 주위로 모든 샤드를 전달해야 합니다. 그림 1은 예를 보여줍니다. 8개 디바이스 각각은 배열의 1/8로 시작하고 모든 복사본으로 끝납니다. 이를 수행하는 효율적인 방법 중 하나는 각 디바이스가 샤딩 차원 링 주위로 샤드를 한 방향 또는 양방향으로 전달하는 것입니다. 한 방향으로 하면, 링크당 $$\text{total size} / N$$ 크기의 $$N - 1$$ hop이 걸리고, 그렇지 않으면 링크당 $$2 \cdot \text{total size} / N$$ 크기의 $\lceil \frac{N}{2} \rceil$ hop이 걸립니다.

{% include figure.liquid path="assets/img/all-gather.gif" %}

**이것이 얼마나 오래 걸릴까요?** 양방향 AllGather를 가져와서 얼마나 걸리는지 계산해봅시다. $$V$$를 배열의 바이트 수라 하고, $$\lvert X\rvert$$를 contracting dimension의 샤드 수라 하겠습니다. 그러면 위 다이어그램에서, 각 hop은 각 방향으로 $V / \lvert X\rvert$ 바이트를 보내므로, 각 hop은 다음이 걸립니다

$$T_{hop} = \frac{2 \cdot V}{|X| \cdot W_\text{ICI}}$$

여기서 $$W_\text{ICI}$$는 **양방향** ICI 대역폭입니다.<d-footnote>분자의 2는 양방향 대역폭을 사용한다는 사실에서 나옵니다. 각 방향으로 $V / |X|$를 보내거나, 총 $2V / |X|$를 보냅니다.</d-footnote> 모든 TPU에 도달하기 위해 총 $\lvert X\rvert / 2$ hop을 보내야 하므로<d-footnote>기술적으로는 $\lceil | X | / 2 \rceil$</d-footnote>, 총 축소는 다음이 걸립니다

$$T_{total} = \frac{2 \cdot V \cdot |X|}{2 \cdot |X| \cdot W_\text{ICI}}$$

$$T_{total} = \frac{V}{W_\text{ICI}}$$

이것이 **$$\lvert X\rvert$$에 의존하지 않는다**는 점에 주목하세요! 그것은 약간 놀랍습니다. TPU가 로컬에서만 연결되어 있음에도 불구하고, 연결의 지역성은 중요하지 않기 때문입니다. 우리는 단지 각 링크의 속도에 의해 병목될 뿐입니다.

<p markdown=1 class="takeaway">**핵심:** 처리량 제한 체제에서 AllGather(또는 ReduceScatter 또는 AllReduce)를 수행할 때, 실제 통신 시간은 배열의 크기와 사용 가능한 대역폭에만 의존하며, 배열이 샤딩된 디바이스 수에는 의존하지 않습니다!</p>

**ICI 지연 시간에 대한 참고:** ICI 링크를 통한 각 hop은 데이터 볼륨과 관계없이 일부 본질적인 오버헤드를 가집니다. 이는 일반적으로 약 1us입니다. 이는 배열 $$A$$가 매우 작고 각 hop이 1us 미만이 걸릴 때, 계산이 실제로 $$\lvert X \rvert$$에 의존하는 "지연 시간 제한" 체제에 들어갈 수 있음을 의미합니다.

{% details 전체 세부사항은 여기를 클릭하세요. %}

$$T_\text{min}$$을 단일 hop의 최소 시간이라 하겠습니다. 그러면

$$T_{hop} = \max \left[ T_{min}, \frac{2 \cdot V}{|X| \cdot W_\text{ICI}} \right]$$

$$T_{total} = \max \left[ \frac{T_{min} \cdot |X|}{2}, \frac{V}{W_\text{ICI}} \right]$$

$$\lvert X \rvert / 2$$ hop을 수행하므로. 큰 reduction이나 gather의 경우, 우리는 확실히 대역폭 제한입니다. 각 hop의 오버헤드가 본질적으로 무시할 수 있을 정도로 많은 데이터를 보내고 있습니다. 하지만 작은 배열의 경우 (예: 모델에서 샘플링할 때), 이는 무시할 수 없으며, ICI 대역폭은 관련이 없습니다. 우리는 순전히 지연 시간에 의해 제한됩니다. 이를 표현하는 다른 방법은 특정 TPU, 예를 들어 `4.5e10` 단방향 ICI 대역폭을 가진 TPU v5e가 주어졌을 때, `4.5e10 * 1e-6 = 45kB` 미만의 버퍼를 보내는 것은 지연 시간 제한이 될 것이라는 것입니다.

{% enddetails %}

**여러 축에 걸쳐 AllGather할 때 무슨 일이 일어날까요?** 여러 축에 걸쳐 gather할 때, gather를 수행할 여러 ICI 차원을 가집니다. 예를 들어, AllGather<sub>XY</sub>([B, D<sub>XY</sub>])는 두 하드웨어 mesh 축에 걸쳐 작동합니다. 이는 사용 가능한 대역폭을 $$n_\text{axes}$$ 인수만큼 증가시킵니다.

{% details 전체 세부사항은 여기를 클릭하세요. %}

일반적으로 다음을 가집니다

$$T_{total} = \max \left[ \frac{T_{min} \cdot \sum_{i} |X_i|}{2}, \frac{V}{W_\text{ICI} \cdot n_\text{axes}} \right]$$

여기서 $$\sum_i \lvert X_i \rvert / 2$$는 TPU mesh에서 가장 긴 경로의 길이입니다.

{% enddetails %}

<b markdown=1 style="color:rgb(144, 92, 255);">퀴즈 2 [AllGather 시간]:</b> [Part 2](../tpus-kr)의 숫자를 사용하여, 2D mesh `{'X': 8, 'Y': 4}`, $$E = 2048$$, $$F = 8192$$를 bfloat16으로 가진 TPUv5e에서 AllGather<sub>Y</sub>([E<sub>Y</sub>, F]) → [E, F]를 수행하는 데 얼마나 걸릴까요? $$E=256, F=256$$의 경우는 어떨까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 몇 가지 기본 양을 계산하는 것부터 시작해봅시다:

1) TPU v5e는 2개 축 각각에 대해 4.5e10 bytes/s의 단방향 ICI 대역폭을 가집니다.
2) (a)의 bfloat16에서, $A[E_Y, F]$를 가지므로 각 디바이스는 bfloat16[512, 8192] 모양의 배열을 보유하며 이는 512 * 8192 * 2 = 8.4MB입니다. 전체 배열의 크기는 2048 * 8192 * 2 = 34MB입니다.

*부분 (1)의 경우*, 위의 공식을 사용할 수 있습니다. 한 축에 걸쳐 AllGather를 수행하므로, $T_{\text{comms}} = 34e6 / 9e10 = 377\mu s$를 가집니다. 지연 시간 제한이 아닌지 확인하기 위해, 크기 4의 축에 걸쳐 최대 3 hop을 가질 것이므로, 지연 시간 제한은 3us 정도이므로 가깝지 않습니다. 하지만, TPU v5e는 한 축의 크기가 16일 때만 wraparound 연결을 가지므로, 여기서 *실제로는 완전한 양방향 AllGather를 할 수 없습니다*. 가장자리에서 다른 가장자리에 도달하기 위해 데이터에 대해 3 hop을 가져야 하므로, 이론적으로 $T_{\text{comms}} = 3 * 8.4e6 / 4.5e10 = 560\mu s$에 가깝습니다. [**여기는**](https://imgur.com/a/RkvpRGQ) **실제 프로파일**이고 [이 Colab](https://colab.research.google.com/drive/15tDZMfNqm2vJjvSzw5VC9qtSwc5td-oV?usp=sharing)에서 $680 \mu s$를 보여주는데, 이론적 대역폭의 100%를 얻지 못할 가능성이 있으므로 합리적입니다! *부분 (2)의 경우* 각 샤드의 크기는 `64 * 256 * 2 = 32kB. 32e3 / 4.5e10 = 0.7us`이므로, 지연 시간 제한입니다. 3 hop을 가지므로, 이는 대략 3 * 1us = 3us가 걸릴 것입니다. [실제로는 8us에 가깝습니다.](https://imgur.com/a/HZLQmYs)

{% enddetails %}

### Case 3: 두 피승수 모두 샤딩된 contracting dimension을 가지는 경우 

세 번째 기본 경우는 두 피승수 모두 같은 mesh 축을 따라 contracting dimension에서 샤딩된 경우입니다:

$$\textbf{A}[I, J_X] \cdot \textbf{B}[J_X, K] \rightarrow C[I, K]$$

이 경우 *로컬* 샤딩된 블록 행렬 곱셈은 같은 contracting 인덱스 집합을 공유하므로 적어도 수행이 *가능*합니다. 하지만 각 곱은 전체 원하는 곱의 *부분합*만을 나타낼 것이고, **X** 차원을 따른 각 디바이스는 이 최종 원하는 곱의 다른 *부분합*으로 남을 것입니다. 이는 너무 일반적이어서 이 조건을 명시적으로 표시하기 위해 표기법을 확장합니다:

$$\textbf{A}[I, J_X] \cdot_\text{LOCAL} \textbf{B}[J_X, K] \rightarrow C[I, K] \{\ U_X \}$$

표기법 **{ U<sub>X</sub> }**는 "**X mesh 축을 따라 unreduced**"라고 읽으며 연산이 어떤 의미에서 "불완전한" 상태를 의미합니다. 최종 합을 기다리고 있어야만 완료될 것이기 때문입니다. $\cdot_\text{LOCAL}$ 구문은 로컬 합을 수행하지만 결과를 unreduced로 남겨둔다는 의미입니다.

이는 행렬 곱셈과 외적에 대한 다음 결과로 볼 수 있습니다:

$$A \cdot B = \sum_{i=1}^{P} \underbrace{A_{:,i} \otimes B_{i,:}}_{\in \mathbb{R}^{n \times m}}$$

여기서 ⊗는 외적입니다. 따라서, 축 **X**의 TPU **i**가 **A**의 **i**번째 열과 **B**의 **i**번째 행을 가진다면, $$A_{:,i} \otimes B_{i,:} \in \mathbb{R}_{n\times m}$$을 얻기 위해 로컬 행렬 곱셈을 할 수 있습니다. 이 행렬은 각 항목에서 **A • B**가 그 항목에서 가진 합의 **i**번째 항을 가집니다. 우리는 여전히 **X** mesh 축에 걸쳐 샤딩된 **P**에 대한 그 합을 수행하여 전체 **A • B**를 얻어야 합니다. 이는 **A**와 **B**를 블록(즉, 샤드)으로 쓴 다음 결과의 각 결과 샤드에 대해 합하는 경우에도 같은 방식으로 작동합니다.

이를 해결하기 위해 **X** 축에 걸친 전체 **AllReduce**를 사용하여 이 합계를 수행할 수 있습니다:

$$\begin{align*}
A[I, J_X] \cdot_\text{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{AllReduce}_X C[I, K] \{ U_X \} \rightarrow &\ C[I, K]
\end{align*}$$

AllReduce는 부분합을 제거하여, 축을 따른 *각* 디바이스가 같은 완전히 합된 값을 가지게 합니다. AllReduce는 이 섹션에서 논의할 여러 핵심 통신 중 두 번째입니다. 첫 번째는 AllGather이고, 다른 것들은 ReduceScatter와 AllToAll입니다. AllReduce는 unreduced(부분적으로 합된) 축을 가진 배열을 가져와서 그 샤드들을 unreduced 축 주위로 전달하고 결과를 누적하여 합을 수행합니다. 시그니처는 다음과 같습니다

$$\textbf{AllReduce}_Y A[I_X, J] \{U_Y\} \rightarrow A[I_X, J]$$

이는 단순히 $\\{U_Y\\}$ 접미사를 제거하지만 그렇지 않으면 결과를 변경하지 않은 채로 둔다는 의미입니다.

**AllReduce는 얼마나 비쌀까요?** AllReduce가 어떻게 수행되는지에 대한 한 가지 정신 모델은 모든 디바이스가 자신의 샤드를 이웃에게 보내고, 받는 모든 샤드를 합한다는 것입니다. 분명히, 각 "샤드"가 전체 배열과 같은 모양을 가지므로 이는 AllGather보다 더 비쌉니다. 일반적으로, **AllReduce는 AllGather보다 2배 비쌉니다.** 이를 보는 한 가지 방법은 **AllReduce**가 두 다른 primitive의 조합으로 표현될 수 있다는 점에 주목하는 것입니다: **ReduceScatter**와 **AllGather**. AllReduce와 같이, ReduceScatter는 배열의 부분합을 해결하지만 주어진 차원을 따라 'scattered' 또는 분할된 출력을 결과로 합니다. AllGather는 그 모든 조각들을 수집하고 주어진 물리적 축을 따라 논리적 축을 'unpartitions/unshards/replicates'합니다.

$$\begin{align*}
\textbf{ReduceScatter}_{Y,J} : A[I_X,J] \{U_Y\} \rightarrow &\ A[I_X, J_Y] \\
\textbf{AllGather}_Y : A[I_X, J_Y] \rightarrow &\ A[I_X, J]
\end{align*}$$

**ReduceScatter는 어떨까요?** AllReduce가 하첨자를 제거하는 것처럼 ($F_Y \to F$ 위), ReduceScatter는 unreduced/부분적으로 합된 배열을 합하고 다른 논리적 축을 같은 mesh 축을 따라 샤딩(scatter)합니다. $[F]\\{U_Y\\} \to [F_Y]$. 애니메이션은 이것이 어떻게 수행되는지 보여줍니다: AllGather와 매우 유사하지만 각 샤드를 유지하는 대신 함께 합한다는 점에 주목하세요. 따라서, 축소를 수행하는 데 걸리는 시간을 제외하면 지연 시간은 대략 같습니다.

{% include figure.liquid path="assets/img/reduce-scatter.gif" class="img-fluid" %}

각 hop의 통신 시간은 AllGather에서와 같이 단순히 샤드당 바이트 $$V$$를 대역폭으로 나눈 것이므로, 다음을 가집니다

$$T_{\text{comms per AllGather or ReduceScatter}} = \frac{V}{W_\text{ICI}}$$

$$T_{\text{comms per AllReduce}} = 2 \cdot \frac{V}{W_\text{ICI}}$$

여기서 $$W_\text{ICI}$$는 양방향 대역폭이며, 축소할 전체 링을 가지고 있는 한입니다.

### Case 4: 두 피승수 모두 같은 축을 따라 샤딩된 non-contracting dimension을 가지는 경우

각 mesh 차원은 tensor를 샤딩할 때 최대 한 번만 나타날 수 있습니다. 위의 규칙들을 수행하면 때때로 이 규칙이 위반되는 상황이 발생할 수 있습니다:

$$A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$$

주어진 샤드, 예를 들어 차원 **X**를 따른 **i**가 **C**의 **(i, i)**번째 샤드, 즉 대각선 항목을 가질 것이므로 이는 유효하지 않습니다. 그러면 모든 샤드 중에 결과의 대각선 항목 외에는 아무것도 복구할 충분한 정보가 없으므로, 이 샤딩을 허용할 수 없습니다.

이를 해결하는 방법은 일부 차원을 AllGather하는 것입니다. 여기서 두 가지 선택이 있습니다:

$$\begin{align*}
\textbf{AllGather}_X A[I_X, J] \rightarrow &\ A[I, J] \\
A[I, J] \cdot B[J, K_X] \rightarrow &\ C[I, K_X]
\end{align*}$$

또는

$$\begin{align*}
\textbf{AllGather}_X B[J, K_X] \rightarrow &\ B[J, K] \\
A[I_X, J] \cdot B[J, K] \rightarrow &\ C[I_X, K]
\end{align*}$$

어느 경우든, 결과는 그 모양에서 **X**를 한 번만 언급할 것입니다. 어느 것을 선택할지는 다음 연산들이 필요로 하는 샤딩에 기반할 것입니다.

## TPU 통신 Primitives에 대한 심층 탐구

이전 4가지 경우는 샤딩된 행렬 곱셈을 수행하는 데 사용되는 여러 "핵심 통신 primitive"를 도입했습니다:

1. **AllGather:** 샤딩에서 하첨자를 제거하여, 샤드를 수집합니다.
2. **ReduceScatter:** 그 축에 걸쳐 샤드를 합하여 배열에서 "un-reduced" 접미사를 제거하고, 배열을 두 번째 축에 걸쳐 샤딩된 채로 남겨둡니다.
3. **AllReduce:** "un-reduced" 접미사를 제거하고, 배열을 그 축을 따라 샤딩되지 않은 채로 남겨둡니다.

Mixture of Experts (MoE) 모델과 다른 계산의 경우에 발생하는 언급할 가치가 있는 한 가지 더 핵심 통신 primitive가 있습니다: **AllToAll**.

### 마지막 통신 primitive: AllToAll

샤딩된 행렬 곱셈을 고려할 때 자연스럽게 발생하지 않지만 실제로 지속적으로 나타나는 최종 기본 collective는 **AllToAll** collective, 또는 더 정확히는 *샤딩된 전치* 또는 resharding 연산의 특별한 경우입니다. 예:

$$\textbf{AllToAll}_{X, J} A[I_X, J] \rightarrow A[I, J_X]$$

AllToAll은 일반적으로 호환되지 않는 레이아웃 방식을 가진 샤딩된 계산의 다른 영역 사이에서 샤딩된 레이아웃을 재배열하는 데 필요합니다. 이들은 샤딩된 mixture-of-experts 모델을 고려할 때 자연스럽게 발생합니다. *AllToAll를 한 축에서 다른 축으로 하첨자를 이동하는 것으로 생각할 수 있습니다.* AllToAll은 각 샤드의 모든 데이터를 링 주위로 복제할 필요가 없으므로, 실제로 AllGather보다 *더 저렴*합니다 (¼배).<d-footnote>짝수 크기 양방향 링의 경우, 각 디바이스는 오른쪽으로 $(N/2 + (N/2-1) + … + 1)$ 청크를, 왼쪽으로 $((N/2-1) + … + 1)$ 청크를 보냅니다 $= 0.5 \cdot (N / 2) \cdot (N/2 + 1) + 0.5 \cdot (N / 2) \cdot (N/2 - 1) = N^2/4$. 각 청크(즉, 샤드의 샤드)의 크기는 $\text{bytes} / N^2$이므로 디바이스당 비용은 $(\text{bytes} / N^2) \cdot N^2 / 4 = \text{bytes} / 4$입니다. 이 결과는 총 대역폭이 디바이스 수에 따라 스케일링되므로 모든 디바이스에 걸쳐 스케일링됩니다.</d-footnote>

{% include figure.liquid path="assets/img/all-to-all.gif" class="img-fluid" %}

### ReduceScatter에 대한 자세한 내용

ReduceScatter는 처음 보이는 것보다 더 기본적인 연산입니다. 실제로 AllGather의 미분이고, 그 역도 마찬가지이기 때문입니다. 즉, forward pass에서 다음을 가진다면:

$$\textbf{AllGather}_X A[I_X] \rightarrow A[I]$$

그러면 reverse-mode 도함수 **A'** (일반적으로 각 샤드에서 다를 것)를 ReduceScatter하여 샤딩된 **A'**를 도출합니다:

$$\textbf{ReduceScatter}_X A'[I] \{ U_X \} \rightarrow A'[I_X]$$

마찬가지로, forward pass에서 $$\text{ReduceScatter}_X(A[I] \{U_X\}) \to A[I_X])$$는 backwards pass에서 $$\text{AllGather}_{X}(A'[I_X]) \to A'[I]$$를 의미합니다.

AllReduce를 AllGather와 ReduceScatter로 바꾸는 것은 또한 최종 AllGather를 나중의 어떤 순간까지 연기할 수 있다는 편리한 속성을 가집니다. 매우 일반적으로 우리는 디바이스에 걸쳐 복제된 전체 행렬 곱의 재조립 비용을 지불하고 싶지 않습니다. 오히려 샤딩된 contracting dimension을 가진 두 피승수를 결합하는 이 경우에도 샤딩된 상태를 보존하고 싶습니다:

$$A[I, J_X] \cdot B[J_X, K] \rightarrow C[I, K_X]$$

이 경우, AllReduce 대신 ReduceScatter를 수행하고, 나중에 선택적으로 AllGather를 수행할 수도 있습니다:

$$\begin{align*}
A[I, J_X] \cdot_{LOCAL} B[J_X, K] \rightarrow &\ C[I, K] \{ U_X \} \\
\textbf{ReduceScatter}_{X,K} C[I, K] \{ U_X \} \rightarrow &\ C[I, K_X]
\end{align*}$$

ReduceScatter는 샤딩된 차원을 *도입*하므로, 이 경우 **I** 또는 **K** 명명된 차원 중 어느 것을 따라 샤딩할지 선택할 자연스러운 자유가 있다는 점에 주목하세요. ReduceScatter를 사용할 때 일반적으로 새로운 샤딩을 도입할 *어떤* 명명된 차원을 선택해야 합니다 (선택은 보통 더 큰 모델링 맥락에 의해 강제되지만). 이것이 우리가 샤딩할 축을 지정하기 위해 **ReduceScatter<sub>X,K</sub>** 구문을 사용하는 이유입니다.

## 우리가 배운 것

* 배열의 샤딩은 TPU mesh의 물리적, 하드웨어 축을 명명하는 **Mesh**와 배열의 논리적 축에 mesh 축 이름을 할당하는 **Sharding**에 의해 지정됩니다.
  * 예를 들어, **A**[I<sub>XY</sub>, J]는 첫 번째 차원이 두 mesh 축 X와 Y를 따라 샤딩된 추상 배열 **A**를 설명합니다. Mesh(mesh_shape=(4, 8), axis_names=('X', 'Y')) 또는 줄임 Mesh({'X': 4, 'Y': 8})와 결합하면, 이는 우리 배열이 첫 번째 차원을 따라 32가지 방법으로 샤딩되었다고 알려줍니다.

* **샤딩된 배열로의 산술은 샤딩된 축을 따라 축약을 수행하지 않는 한 샤딩되지 않은 배열과 정확히 같이 작동합니다**. 그 경우, 일부 통신을 도입해야 합니다. 우리는 4가지 경우를 고려합니다:

  1. *어떤 배열도 contracting dimension을 따라 샤딩되지 않은 경우*: 통신이 필요하지 않습니다.
  2. *한 배열이 contracting dimension을 따라 샤딩된 경우* (또는 contracting dimension이 다른 축을 따라 샤딩된 경우): 연산을 수행하기 전에 입력 중 하나를 AllGather합니다.
  3. *두 배열 모두 contracting dimension을 따라 동일하게 샤딩된 경우:* 샤드를 로컬로 곱한 다음 AllReduce 또는 ReduceScatter를 수행합니다.
  4. *두 배열 모두 non-contracting dimension을 따라 같은 mesh 축을 따라 샤딩된 경우:* 먼저 입력 중 하나를 AllGather합니다.

* TPU는 대략 **4가지 핵심 통신 primitive**를 사용합니다:
  1. AllGather: $[A_X, B] \to [A, B]$
  2. ReduceScatter: $[A, B] \\{U_X\\} \to [A, B_X]$
  3. AllToAll: $[A, B_X] \to [A_X, B]$
  4. AllReduce: $[A_X, B]\\{U_Y\\} \to [A_X, B]$ (기술적으로는 ReduceScatter + AllGather를 결합하므로 primitive가 아님)

{% include figure.liquid path="assets/img/all-collectives.png" class="img-fluid" %}

* 이러한 각 연산의 비용과 지연 시간은 **축의 크기에 의존하지 않지만 (대역폭 제한인 한), 입력 배열의 크기와 링크의 대역폭에만 의존합니다**. 단방향 AllGather/ReduceScatter의 경우:

$$T_{\text{comm per AllGather or ReduceScatter}} = \frac{\text{Data volume}}{\text{bandwidth}} \cdot \frac{\text{Axis} - 1}{\text{Axis}}
\longrightarrow \frac{\text{Data volume}}{\text{bandwidth (bidirectional)}}$$

* AllReduce는 ReduceScatter 다음에 AllGather로 구성되므로, 위 비용의 2배를 가집니다. AllToAll은 샤드를 링 주위로 부분적으로만 전달하면 되므로 AllGather 비용의 ¼입니다. 다음은 요약입니다:

| Operation         | Description                                                                                                        | Syntax                           | Runtime                                          |
| :---------------- | :----------------------------------------------------------------------------------------------------------------- | :------------------------------- | :----------------------------------------------- |
| **AllGather**     | 샤딩된 배열의 모든 샤드를 축을 따라 수집하여, 하첨자를 제거합니다.                                     | $[A_X, B] \to [A, B]$            | bytes / (bidirectional ICI bandwidth * num_axes) |
| **ReduceScatter** | 부분적으로 합된 배열을 축을 따라 합하고 다른 축을 따라 샤딩합니다 (하첨자 추가).                 | $[A, B] \\{U_X\\} \to [A_X, B]$  | AllGather와 같음                                |
| **AllReduce**     | 부분적으로 합된 배열을 축을 따라 합합니다. { U<sub>x</sub> }를 제거합니다. AllGather와 ReduceScatter를 결합합니다. | $[A_X, B]\\{U_Y\\} \to [A_X, B]$ | 2 * AllGather                                    |
| **AllToAll**      | 축을 수집(복제)하고 다른 차원을 같은 축을 따라 샤딩합니다.                                 | $[A, B_X] \to [A_X, B]$          | 양방향 링의 경우 AllGather / 4           |

## 연습할 문제들

*이 섹션의 내용을 기반으로 한 교육적인 문제들입니다. 현재로서는 모든 답을 포함하지 않지만 가능한 한 더 많은 답을 작성할 것입니다.*

**문제 1 [복제된 샤딩]**: 배열이 $A[I_X, J, K, \ldots]$ (즉, $X$에 걸쳐서만 샤딩됨)로 샤딩되고, mesh `Mesh({'X': 4, 'Y': 8, 'Z': 2})`를 가집니다. 모든 칩에 걸쳐 $A$가 차지하는 총 바이트 수와 배열의 한 복사본 크기의 비율은 얼마입니까?

{% details 답을 보려면 여기를 클릭하세요. %}

우리 배열은 크기 4를 가진 X를 따라서만 샤딩되므로, 효과적으로 각 샤드의 크기는 $[I / 4, J, K, \ldots] = \text{sizeof}(A) / 4$입니다. 배열이 Y와 Z에 걸쳐 복제되므로, 총 크기는 $Y \cdot Z \cdot \text{sizeof}(A)$이므로, 총 크기 대 단일 칩 크기의 비율은 $Y \cdot Z \cdot \text{sizeof}(A) / \text{sizeof}(A) = 16$입니다.

{% enddetails %}

**문제 2 [AllGather 지연 시간]**: mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})`를 가진 TPUv4p 4x4x4 slice에서 $B=1024$이고 $D=4096$을 bfloat16으로 할 때 $\text{AllGather}_X([B_X, D_Y])$는 얼마나 걸려야 할까요? $$\text{AllGather}_{XY}([B_X, D_Y])$$는 어떨까요? $$\text{AllReduce}_Z([B_X, D_Y] \{U_Z \})$$는 어떨까요?

{% details 답을 보려면 여기를 클릭하세요. %}

모든 축에서 wraparound 링크를 가진 전체 `4x4x4` 큐브를 가지므로, 9e10 양방향 대역폭을 사용할 수 있습니다.

1. 한 축에 걸쳐서만 gather하고 다른 축이 샤딩되어 있으므로, 1개 축에 걸쳐 $2BD / Y$ 바이트를 효과적으로 gather하고 있습니다. TPU v4p의 ICI 대역폭이 9e10 bytes/second 양방향이므로, 이는 $2BD / (9e10 \cdot Y) = 2 \cdot 1024 \cdot 4096 / (9e10 \cdot 4) = 23 \mu s$가 걸립니다.

2. 이전보다 2배의 대역폭을 가지지만 전체 배열을 AllGather하고 있으므로, `T = 2BD / (2 * W) = 2*1024*4096 / (2 * 9e10) = 46us`입니다. 이는 4us (hop당 1us)의 지연 시간 제한에서 멀므로, 괜찮습니다.

3. AllReduce의 비용은 AllGather의 2배입니다. 각 샤드의 크기는 $2BD / (X * Y)$이므로, 비용은 약 $4BD / (X * Y * W)$, 즉 대략 `4 * 1024 * 4096 / (16 * 9e10) = 11.6us`입니다.

{% enddetails %}

**문제 3 [지연 시간 제한 AllGather]**: $B$가 매우 작은 (128이라고 하자) $\text{AllGather}_X([B_X])$를 수행한다고 가정합시다. mesh `Mesh({'X': 4, 'Y': 4, 'Z': 4})`를 가진 TPUv4p 4x4x4 slice에서 bfloat16으로 이것이 얼마나 걸려야 할까요? *힌트: 아마 지연 시간 제한일 것입니다.*

{% details 답을 보려면 여기를 클릭하세요. %}

bfloat16의 배열은 총 256바이트만 사용하고, 디바이스당 64바이트만 사용합니다. TPU v4p에서 크기 4의 축을 가지므로, wraparound 링크를 가지고 있어서 배열을 양방향으로 보낼 수 있습니다. `4.5e10`의 단방향 대역폭으로, 각 hop은 대략 `64 / 4.5e10 ~ 0`이 걸리므로, 확실히 지연 시간 제한입니다. hop 수를 세면, 단 2 hop으로 전체 gather를 수행할 수 있으므로, 대략 2us가 좋은 추정입니다.

{% enddetails %}

**문제 4 [matmul 전략]**: $X[B, D] \cdot_D Y[D_X, F] \to Z[B, F]$를 수행하기 위해, 이 섹션에서는 $\text{AllGather}_X(Y[D_X, F])$를 수행하고 완전히 복제된 행렬을 곱하라고 알려줍니다 (Case 2, *전략 1*). 대신, $X[B, D_X] \cdot_D Y[D_X, F] \to Z[B, F] \\{U_X\\}$ (Case 4, *전략 2*)처럼 로컬 샤드를 곱한 다음, $\text{AllReduce}_X(Z[B, F] \\{ U_X\\})$를 할 수도 있습니다. 이들 각각은 얼마나 많은 FLOP과 통신을 수행하나요? 어느 것이 더 좋고 왜 그럴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

기준선(*전략 1*)부터 시작해봅시다. 보여준 바와 같이, AllGather의 비용은 $2DF / W_\text{ici}$입니다. 완전히 복제된 배열을 가지면, 총 계산 시간은 $2BDF / C$입니다 (여기서 $C$는 accelerator FLOP/s이며, 각 TPU가 같은 FLOP을 수행하므로). 따라서 다음을 가집니다

$$T_\text{total (Strategy 1)} = \max\left(\frac{2BDF}{C}, \frac{2DF}{W_\text{ici}}\right)$$

비교해서, 새로운 전략(전략 2)은 $2BF$ 바이트에 대해 AllReduce를 수행하는데, 이는 $4BF / W_\text{ici}$ 비용을 가지지만 $1 / X$ 더 적은 FLOP을 수행합니다 (계산이 샤딩되므로). 이는 $2\cdot B\cdot D\cdot F / X$ FLOP을 수행하고 결과 AllReduce는 bfloat16에서 $$2 \cdot 2 \cdot B \cdot F$$ 바이트를 통신함을 의미합니다. 따라서, *전략 2*의 총 시간 (AllGather 없이, 나중에 AllReduce만)은 대략 다음과 같습니다

$$T_\text{total} = \max\left(\frac{2BDF}{X \cdot C}, \frac{4BF}{W_\text{ici}}\right)$$

질문은: *이들 중 어느 것이 더 클까요?* 전략 (2)는 $D / (X \cdot C) > 2 / W_\text{ici}$일 때, 즉 $D / 2X > C / W_\text{ici} \approx 2550 \rightarrow X < D / (2 * 2550)$일 때 compute bound입니다. $D \approx 8k$를 합리적으로 예상할 수 있으므로, 이는 대략 $X < 2$를 의미하는데 이는 불가능합니다 – 따라서 전략 2에서는 기본적으로 항상 통신 제한입니다. 기준선(전략 1)에서는 $$B < C / W_\text{ici} = 2550$$일 때 통신 제한이며 이는 종종 (항상은 아니지만) 참입니다.

따라서 $B < 2550$이면, 두 경우 모두 통신 제한이고 다음을 가집니다

$$T_\text{comms for Strategy 2} < T_\text{comms for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2DF}{W_\text{ici}}$$

이는 $D > 2B$일 때 참이며 $2B < 5100$입니다. 이는 종종 참이므로, 배치가 작으면 전략 2가 때때로 더 좋을 수 있습니다. 배치가 클 때 ($B > 2550$), 다음을 가집니다

$$T_\text{comms for Strategy 2} < T_\text{math for Strategy 1} \Leftrightarrow \frac{4BF}{W_\text{ici}} < \frac{2BDF}{C}$$

이는 $2 / W_\text{ici} < D / C$일 때, 즉 $D > 2 * 2550 = 5100$일 때 참이며, 이는 대형 모델에 대해 보통 참입니다. 따라서 이 대안 전략은 $D$가 작지 않은 한 일반적으로 대형 모델에 더 좋습니다.

*왜 항상 이렇게 하지 않을까요?* 실제로는 때때로 이렇게 할 수도 있지만, matmul에 대한 입력 중 하나의 contracting dimension이 다른 입력이 샤딩되지 않은 축을 따라 샤딩되는 것은 일반적으로 드뭅니다. 예를 들어, FSDP ([섹션 5](../training-kr)에서 설명)를 하고 있다면, 매개변수를 데이터 차원에 걸쳐 샤딩하지만 activation도 _데이터를 따라 샤딩될_ 것입니다. 따라서 이런 의미에서 이는 많이 나타나지 않습니다.

{% enddetails %}

**문제 5 [최소 지연 시간]**: TPUv5p 4x4x4에서 가능한 한 낮은 지연 시간으로 matmul $A[B, D] \cdot_D B[D, F] \to C[B, F]$를 하고 싶다고 가정합시다. 입력이 어떻게 샤딩되어야 할까요? 총 FLOP과 통신 시간은 얼마입니까?

**문제 6:** TPUv5e 4x4에서 $A[I_X, J_Y] \cdot_J B[J_Y, K] \to C[I_X, K]$를 수행하고 싶다고 가정합시다. 어떤 통신을 수행합니까? 통신 대 계산에 얼마나 많은 시간을 보내나요?

* $A[I_X, J] \cdot_J B[J_X, K_Y] \to C[I_X, K_Y]$는 어떨까요? 이는 데이터, tensor, zero 샤딩을 결합하는 훈련의 가장 표준적인 설정입니다.
* $A[I_X, J] \cdot_J B[J, K_Y] \to C[I_X, K_Y]$는 어떨까요? 이는 순수 tensor 병렬성(+데이터)을 하는 추론의 표준입니다.

**문제 7:** 일반적인 Transformer 블록은 $F \gg D$인 두 행렬 $B[D, F]$와 $C[F, D]$를 가집니다. 배치 크기 B로, 전체 블록은 $$x[B, D]$$와 함께 $$C \cdot B \cdot x$$입니다. $$D=8192$$, $$F=32768$$, $$B=128$$을 선택하고 모든 것이 bfloat16이라고 가정합시다. TPUv5e 2x2 slice에서 실행하고 있지만 각 TPU가 300MB의 여유 메모리만 가진다고 가정합시다. 메모리 한계 아래로 유지하면서 전체 시간을 최소화하기 위해 **B, C, 출력이 어떻게 샤딩되어야 할까요? 통신과 FLOP에 얼마나 많은 시간을 보내나요?**

**문제 8 [도전]**: 위의 짧은 코드 스니펫을 템플릿으로 사용하여, 샤딩된 배열을 할당하고 pmap 또는 shard_map을 사용하여 4가지 주요 통신 primitive (AllGather, AllReduce, ReduceScatter, AllToAll) 각각을 벤치마크하세요. `jax.lax.all_gather`, `jax.lax.psum`, `jax.lax.psum_scatter`, `jax.lax.all_to_all`을 사용하고 싶을 것입니다. 이러한 함수들의 의미를 이해하나요? 얼마나 걸리나요?

**문제 9 [샤딩된 matmul의 또 다른 전략?]**: [위에서](#case-2-한-피승수가-샤딩된-contracting-dimension을-가지는-경우) matmul에 대한 입력 중 하나만 contracting dimension을 따라 샤딩된 경우, 샤딩된 행렬을 AllGather하고 결과 축약을 로컬로 수행해야 한다고 주장했습니다. 생각해볼 수 있는 또 다른 전략은 샤딩된 matmul을 수행한 다음 (두 입력 모두 contracting dimension을 따라 샤딩된 것처럼) 결과를 AllReduce하는 것입니다. 즉, $A[I, J_X] *_J B[J, K] \to C[I, K]$를 다음 방법으로:

1. $C[I, K] \\{ U_X \\} = A[I, J_X] \cdot B[J_X, K]$
2. $C[I, K] = \text{AllReduce}(C[I, K] \\{ U_X\\})$

다음에 답하세요:

1. 행렬 $A[N, M]$과 $B[M, K]$에 대해 이 알고리즘을 명시적으로 작성하고, 인덱스를 사용하여 어떤 디바이스에서 정확히 어떤 계산이 수행되는지 보여주세요. $A$가 ND 디바이스에 걸쳐 $A[I, J_X]$로 샤딩되고, 출력이 모든 디바이스에서 복제되기를 원한다고 가정합니다.
2. 이제 최종 결과가 각 디바이스에서 복제되지 않고 대신 샤딩되어도 괜찮다고 가정합니다 (N 또는 K 차원 중 하나에 걸쳐). 위의 알고리즘이 어떻게 변할까요?
3. 위의 전략 ((a)가 아닌 (b))의 통신 비용만 보면, 이 통신 비용이 먼저 A를 AllGather한 다음 matmul을 하는 알고리즘의 통신 비용과 어떻게 비교됩니까?

{% details 답을 보려면 여기를 클릭하세요. %}

1. 먼저 외적을 계산하여 결과를 $$O[N, K]: o_{kj} = \sum_i a_{ki} b_{ij}$$에 저장합니다. 반복되는 인덱스가 축약되는 것이 아니라는 점에 주목하세요. 외적을 하고 있기 때문입니다. 여기서 합은 우리가 사용하는 특정 디바이스에 저장된 i 값의 집합에 걸쳐 범위가 지정됩니다. 예를 들어, contracting axis의 크기가 16이고 4개의 디바이스가 있다면, 디바이스 0에서 i는 {0, 1, 2, 3}에 걸쳐 범위가 지정되고; 디바이스 1에서 i는 {4, 5, 6, 7}에 걸쳐; 디바이스 2에서 i는 {8, 9, 10, 11}에 걸쳐; 디바이스 3에서 i는 {12, 13, 14, 15}에 걸쳐 범위가 지정됩니다. 그런 다음 각 디바이스에 있는 $O[N, K]$의 부분합을 AllReduce하여 전체 $O[N, K]$를 형성합니다.

2. 2단계에서 AllReduce 대신, 더 저렴한 ReduceScatter를 어느 축을 따라서든 할 수 있습니다: $[N, K] \\{ U_X \\} \to [N_X, K]$ 또는 $[N, K] \\{ U_X \\} \to [N, K_X]$.

3. 위의 본문에서 설명한 바와 같이, (처리량 제한일 때) AllGather를 수행하는 비용은 ReduceScatter의 비용과 같습니다; 단순히 처리하는 전체 행렬의 크기로 주어집니다. 따라서 gather-then-matmul 알고리즘에서, 이는 $NM$에 비례합니다 (우리가 $A$를 $\text{AllGather}$하므로); matmul-then-reduce-scatter 알고리즘에서, 이는 NK에 비례합니다 (우리가 $O$를 reduce-scatter하므로). 따라서 두 알고리즘의 통신 비용 비율은 `M/K`입니다.

{% enddetails %}

**문제 10: AllToAll로 재미있게:** 위의 표에서, (처리량 제한 체제에서) AllToAll을 수행하는 시간이 AllGather 또는 ReduceScatter를 수행하는 시간보다 4배 낮다고 언급되었습니다. 이 문제에서는 그 4배가 어디서 나오는지 보고, 양방향 ICI 링크 대신 단방향 ICI 링크만 가진다면 이 인수가 어떻게 변할지도 볼 것입니다.

1. 먼저 단방향 경우부터 시작합시다. 링 토폴로지에 *D*개의 디바이스가 있고, AllGather 또는 ReduceScatter를 하고 있다면, $A[I_X, J]$로 샤딩된 N x N 행렬 *A*에 대해 ($D$가 $N$을 나눈다고 가정합니다). 이 두 collective에 포함된 통신을 설명하고, 이 알고리즘 전체 동안 **단일** ICI 링크를 통해 전송되는 스칼라(float 또는 int)의 총 수를 계산하세요.

2. 이제 여전히 단방향 ICI 경우에서 AllToAll에 대해 생각해봅시다. 이 경우 알고리즘이 all-gather 경우와 어떻게 다른가요? 이 알고리즘에서 단일 ICI 링크를 통해 전송되는 스칼라의 수를 계산하세요.

3. 부분 (a)와 (b)에 대한 답 사이의 비율이 좋은 숫자여야 합니다. 이 인수가 간단한 용어로 어디서 나오는지 설명하세요.

4. 이제 양방향 통신을 추가해봅시다. 이것이 all-gather 경우에 필요한 총 시간에 어떤 영향을 미칩니까?

5. 양방향 통신을 추가하는 것이 AllToAll 경우에 필요한 총 시간에 어떤 영향을 미칩니까?

6. 이제 양방향 링에서 AllGather 시간과 AllToAll 시간 사이의 비율을 간단히 설명하세요.

{% details 답을 보려면 여기를 클릭하세요. %}

(1) **해답:** 과정은 간단합니다: 알고리즘의 각 단계에서, 각 디바이스는 행렬의 단일 샤드 "스트립"(총 크기 $$\frac{N}{D} \times N$$ 요소)을 가장 가까운 이웃에게 보냅니다. 이는 각 샤드가 시작한 디바이스를 제외한 모든 디바이스로 통신되어야 하므로 $$D-1$$번 발생합니다. 따라서 총 $$\frac{N^2(D-1)}{D}$$ 스칼라가 각 디바이스에 의해, 즉 단일 ICI 링크를 통해 전송됩니다.

**답:** $$N^2 (1-\frac{1}{D})$$, 또는 $$D >> 1$$일 때 단순히 $$N^2$$.

(2) **해답:** 통신 관점에서 AllToAll과 AllGather의 주요 차이점은 AllToAll에서는 특정 디바이스에 있는 샤드의 전체가 다른 모든 디바이스로 통신될 필요가 없다는 것입니다. 특정 디바이스(디바이스 0이라고 하자)에 저장된 샤드가 $$[A, B, C, D]$$ (여기서 A,B,C,D는 행렬이고 설명을 위해 4개 디바이스가 있는 링을 상상하고 있습니다)라고 상상해보세요. 이제 행렬 $$A$$는 어디로도 통신될 필요가 없고, 행렬 $$B$$는 디바이스 1에 도달해야 하고; 행렬 $$C$$는 디바이스 2에 도달하고; 행렬 $$D$$는 디바이스 3에 도달합니다. 따라서 알고리즘의 첫 번째 단계에서, $$B$$, $$C$$, $$D$$를 디바이스 1에 보냅니다; 다음 단계에서, 디바이스 1은 $$C$$와 $$D$$를 계속해서 디바이스 2로 보냅니다; 마지막 단계에서, 디바이스 2는 $$D$$만 디바이스 3으로 보냅니다. 이 경우 전송되는 총 매개변수 수는 $$(\text{A/B/C/D의 크기}) * (3 + 2 + 1)$$입니다. A/B/C/D의 크기는 (이제 일반적인 경우에서) $$\frac{N^2}{D^2}$$이고, 다시 일반적인 경우에서 $$(3 + 2 + 1)$$ 항은 $$((D-1) + (D-2) + … + 1)$$이 되거나, $$\frac{(D)(D-1)}{2}$$입니다. 따라서 단일 ICI 링크를 통해 전송되는 총 바이트 수는 $$\frac{N^2(D-1)}{D \times 2}$$입니다.

**답:** $$\frac{N^2}{2}(1-\frac{1}{D})$$, 또는 $$D >> 1$$일 때 단순히 $$\frac{N^2}{2}$$.

(3) **해답:** 인수는 단순히 $$\frac{1}{2}$$입니다. 즉, AllToAll은 단방향 링 토폴로지에서 all-gather/ReduceScatter의 절반 비용입니다. 위의 도출을 보면, 이는 궁극적으로 all-gather 경우에는 $$(D-1)$$번 각각 같은 크기의 블록을 전송한다는 사실에서 나옵니다. 즉, $$ \text{tiny block size} * (D + D + D + … + D)$$ 합을 하는 반면, AllToAll 경우에는 $$\text{tiny block size} * (D + D-1 + D-2 + … + 1)$$ 합을 하고 있습니다. 따라서 2배는 본질적으로 $$1 + 2 + \ldots + n = n(n+1)/2$$라는 사실에서 나옵니다.

(4) **해답**: 각 "샤딩된 스트립"이 동시에 두 방향으로 보내질 수 있으므로, 어떤 하나의 링크가 운반해야 하는 스칼라의 총 수가 이제 2배 줄어듭니다.

(5) **해답**: 이 경우, 단방향 경우에 비해 4배를 얻습니다. 이는 디바이스 0에서 시작하는 단일 샤딩된 스트립에서 각각의 크기-(N2/D2) 블록의 운명을 고려함으로써 가장 쉽게 볼 수 있습니다. (단방향 경우에서처럼) 이 블록 중 하나를 D-1 거리만큼, 다른 블록을 D-2 거리만큼, 등등 1까지 보내는 대신, 이제 스트립을 오른쪽 또는 왼쪽으로 움직이는 블록으로 나누어 최대 ceil(D/2) 거리를 움직입니다. 따라서 해당하는 합은 이제 $$D/2 + D/2 - 1 + D/2 - 2 + … = D/2 \cdot (D/2+1)/2$$가 되거나, 큰 $$D$$의 한계에서 $$D^2/8$$입니다. 이를 단방향 경우의 $$D^2/2$$와 비교하면, 4배를 얻었다는 것을 알 수 있습니다.

(6) **해답:** 단방향 링에서, AllToAll 시간이 이미 all-gather 시간보다 2배 빨랐습니다; 이는 전체 스트립을 모든 단일 디바이스에 보낼 필요가 없다는 사실에서 나옵니다. 그런 다음, 양방향성을 추가했을 때, AllToAll에 대해서는 4배 향상을, all-gather에 대해서는 2배 향상만을 봤습니다. 이러한 비율을 종합하면, 우리가 찾던 4배를 얻습니다.

{% enddetails %}

<h3 markdown=1 class="next-section">Part 3은 여기까지입니다! Transformer 수학에 대한 Part 4는 [여기를 클릭하세요](../transformers-kr)!</h3>