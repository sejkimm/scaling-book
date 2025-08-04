---
layout: distill
title: "Rooflines에 대한 모든 것"
# permalink: /main/
description: "하드웨어에서 알고리즘을 실행할 때, 우리는 세 가지에 의해 제한됩니다: 컴퓨터가 수학을 얼마나 빨리 할 수 있는지(OPs/second), 데이터를 이동하는 데 사용할 수 있는 대역폭(bytes/second), 그리고 데이터를 저장하는 데 사용할 수 있는 총 메모리(bytes). 이러한 'roofline' 제약 조건을 통해 주어진 계산의 시간에 대한 상한과 하한을 구할 수 있습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 1

previous_section_url: "../index"
previous_section_name: "Part 0: 소개"

next_section_url: ../tpus
next_section_name: "Part 2: TPUs"

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
  
  - name: 시간은 어디로 가는가?
  - subsections:
    - name: "Rooflines 시각화하기"
    - name: "행렬 곱셈"
    - name: "네트워크 통신 rooflines"
  - name: 몇 가지 연습 문제

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

## 시간은 어디로 가는가?

매우 간단한 질문부터 시작해봅시다: *왜 알고리즘이 50s나 5ms가 아닌 50ms가 걸릴까요*? 실제로 모델 내에서 상당한 시간이 걸리는 것은 무엇이고 얼마나 걸릴 것으로 예상해야 할까요?

**계산:** 딥러닝 모델은 본질적으로 부동 소수점 곱셈과 덧셈 '연산'(FLOP)으로 구성된 여러 행렬 곱셈의 집합입니다. 우리의 accelerator 속도가 이를 계산하는 데 걸리는 시간을 결정합니다:

$$\begin{equation}
T_\text{math} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
\end{equation}$$

예를 들어, NVIDIA H100은 약 9.89e14 bfloat16<d-footnote>bf16은 <a href="https://en.wikipedia.org/wiki/Bfloat16_floating-point_format">bfloat16</a>의 줄임말로, ML에서 자주 사용되는 16비트 부동 소수점 형식입니다.</d-footnote> FLOP/s를 수행할 수 있고, TPU v6e는 9.1e14 FLOP/s를 수행할 수 있습니다. 즉, H100에서 1e12 FLOP을 수행하는 데는 (대략) `1e12 / 9.89e14 = 1.01ms`가 걸리고, TPU v6e에서는 `1e12 / 9.1e14 = 1.1ms`가 걸립니다.<d-footnote>이 칩들은 가격이 다르며, 이 비교는 비용으로 정규화하지 않았다는 점에 유의하세요.</d-footnote>

**칩 내 통신:** *Accelerator 내에서*, tensor들은 on-chip 메모리(HBM)와 compute core 사이에서 전송되어야 합니다. 이 링크의 대역폭을 "HBM 대역폭"이라고 합니다<d-footnote>NVIDIA는 이를 "메모리 대역폭"이라고도 부릅니다.</d-footnote>. H100에서 [이는 약 3.35TB/s](https://www.nvidia.com/en-us/data-center/h100/)이고, TPU v6e에서 [이는 약 1.6TB/s](https://cloud.google.com/tpu/docs/v6e)입니다<d-footnote>이들이 광고된 수치이지만, 실제로는 달성하기 어려운 경우가 많습니다. B100의 경우, 광고된 bf16 처리량의 82% 이상을 달성할 수 있는 구현이 거의 없는 반면, TPU v5p는 일반적으로 약 95%를 달성할 수 있습니다.</d-footnote>.

**칩 간 통신:** 모델을 *여러 accelerator에 걸쳐* 분산할 때, tensor들은 종종 그들 사이에서 전송되어야 합니다. 하드웨어에는 종종 이를 위한 몇 가지 옵션(ICI, DCN, PCIe)이 있으며, 각각 다른 대역폭을 가집니다.

통신이 칩 내인지 칩 간인지와 관계없이, 우리는 이를 bytes/s로 측정하고 총 통신 시간을 다음과 같이 추정합니다:

$$\begin{equation}
T_\text{comms} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
\end{equation}$$

일반적으로 (항상은 아니지만), 단일 칩 내의 계산은 칩 내 통신 및 칩 간 통신과 겹칠 수 있습니다. 이는 **통신과 계산 시간의 최댓값을 사용하여 훈련 및 추론 시간의 하한을 구할 수 있다**는 의미입니다. 또한 **그들의 합으로 상한을 구할 수도 있습니다**. 실제로는 대수가 더 간단하고 통신과 계산을 겹쳐서 이 하한에 가깝게 갈 수 있기 때문에 최댓값에 대해 최적화합니다. 최댓값을 염두에 두고 최적화하면 $T_\text{math} + T_\text{comms} \leq 2 * \max(T_\text{math}, T_\text{comms})$이므로 하한과 상한이 최대 2배 차이납니다. 그 다음에는 특정 모델과 대상 시스템을 프로파일링하여 알 수 있는 'overlap regions'과 오버헤드를 모델링하여 정확도를 높입니다.

$$\begin{equation}
T_\text{lower}=\max(T_\text{math}, T_\text{comms})
\end{equation}$$

$$\begin{equation}
T_\text{upper} = T_\text{math} + T_\text{comms}
\end{equation}$$

통신과 계산을 완벽하게 겹칠 수 있다고 가정하면, $T_\text{math} > T_\text{comms}$일 때 하드웨어에서 완전한 활용을 볼 수 있습니다. 이를 "compute-bound"라고 합니다. $T_\text{comms} > T_\text{math}$일 때는 "communication-bound"가 되는 경향이 있으며, accelerator FLOP/s의 적어도 일부가 데이터가 전달되기를 기다리며 낭비됩니다. 연산이 compute-bound인지 communication-bound인지 알 수 있는 한 가지 방법은 "*arithmetic intensity*" 또는 "*operational intensity*"를 보는 것입니다.

**정의:** 알고리즘의 arithmetic intensity는 그것이 수행하는 총 FLOP과 통신해야 하는 바이트 수(칩 내 또는 칩 간)의 비율로 주어집니다.

$$\begin{equation}
\text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
\end{equation}$$

Arithmetic intensity는 주어진 연산의 "FLOP per byte"를 측정합니다. 1차 근사에서, arithmetic intensity가 높을 때 $T_\text{math}$가 $T_\text{comms}$에 비해 크고 일반적으로 사용 가능한 FLOP의 대부분을 사용합니다. 반대의 경우, 통신에 더 많은 시간을 보내고 FLOP을 낭비합니다. 이 전환점은 하드웨어의 "peak arithmetic intensity", 즉 peak accelerator FLOP/s와 accelerator 대역폭의 비율입니다.

$$\begin{align*}
T_\text{math} > T_\text{comms} \Leftrightarrow \frac{\text{Computation FLOPs}} {\text{Accelerator FLOPs/s}} > \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} > \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}} & \\[0.5em]
\Leftrightarrow \text{Intensity}(\text{Computation}) > \text{Intensity}(\text{Accelerator}) & \\
\end{align*}$$

$\text{Intensity}(\text{Accelerator})$ 양은 우리 accelerator가 peak FLOP/s를 달성하는 arithmetic intensity입니다. **TPU v5e MXU의 경우, 이는 약 240 FLOP/byte입니다**<d-footnote>MXU는 TPU의 matrix multiply unit입니다. TPU에는 elementwise 연산을 담당하는 VPU와 같이 다른 peak FLOP/s를 가진 다른 accelerator들이 있기 때문에 여기서 이를 명시합니다.</d-footnote>. TPU는 `1.97e14` FLOP/s를 수행할 수 있고 HBM에서 `8.2e11` bytes/s를 로드할 수 있기 때문입니다. 즉, 알고리즘의 arithmetic intensity가 240<d-footnote>이는 알고리즘이 HBM에서 가중치를 로드하고 MXU에서 실행되는 경우에만 해당됩니다. 다음 섹션에서 논의할 바와 같이, 때로는 훨씬 높은 대역폭을 가진 VMEM에 매개변수를 저장할 수 있습니다. 많은 알고리즘이 또한 다른 성능 특성을 가진 VPU에서 실행됩니다.</d-footnote> FLOP/byte보다 낮으면, 바이트 로딩에 의해 제한되어 하드웨어를 잘 활용하지 못할 것입니다. 이런 예를 하나 살펴봅시다:

**<span style="color:#7ab5ff">예제 (내적)</span>:** bfloat16 정밀도에서 두 벡터의 내적을 계산하려면, `x • y: bf16[N], bf16[N] → bf16[1]`, $x$와 $y$를 메모리에서 로드해야 하는데, 각각 $2 * N = 2N$ 바이트를 가지고, $N$번의 곱셈과 $N-1$번의 덧셈을 수행하고, $2$ 바이트를 HBM에 다시 쓰기합니다

$$\begin{equation}
\text{Intensity}(\text{dot product}) = \frac{\text{Total FLOPs}}{\text{Total Bytes}} = \frac{N + N - 1}{2N + 2N + 2} = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}
\end{equation}$$

$N\rightarrow\infty$일 때. 따라서 내적의 arithmetic intensity는 $\frac{1}{2}$이거나, 다른 말로 하면, 내적은 로드된 바이트당 0.5개의 부동 소수점 연산을 수행합니다. 이는 우리의 arithmetic intensity가 하드웨어의 것보다 낮다는 의미이며 communication-bound가 될 것입니다.<d-footnote>위의 240이라는 숫자는 다음 섹션에서 보게 될 것처럼 내적이 MXU가 아닌 VPU에서 수행되므로 여기서 올바른 비교가 아닙니다. TPU v5p VPU는 초당 대략 7e12 FLOP을 수행할 수 있으므로, 임계 intensity는 약 3이며, 이는 여전히 어느 정도 통신 제한임을 의미합니다. 어느 쪽이든, intensity가 낮고 일정하다는 사실은 대부분의 하드웨어에서 compute-bound가 되기 어렵다는 것을 의미합니다.</d-footnote>

### Rooflines 시각화하기

**Roofline plot**을 사용하여 메모리와 계산 사이의 트레이드오프를 시각화할 수 있습니다. 이는 하드웨어에서 알고리즘의 달성 가능한 최대 FLOP/s(처리량)(y축)를 그 알고리즘의 arithmetic intensity(x축)에 대해 플롯합니다. 다음은 log-log plot의 예입니다:

{% include figure.liquid path="assets/img/roofline-improved.png" class="img-fluid" caption="<b>그림:</b> 서로 다른 arithmetic intensities를 가진 두 알고리즘(Algo 1과 Algo 2)과 서로 다른 대역폭(BW1과 BW2)에서의 해당 이론적 최대 처리량을 보여주는 roofline plot 예제. 빨간색 영역에서 알고리즘은 두 대역폭 모두에서 대역폭 제한을 받으며 하드웨어의 peak FLOP/s 중 일부를 낭비하고 있습니다. 노란색 영역은 낮은 대역폭(BW1)에서만 대역폭 제한을 받습니다. 녹색 영역은 모든 대역폭에서 compute-bound입니다. 여기서는 accelerator의 peak FLOP/s를 사용하고 있으며 대역폭을 증가시키거나 intensity를 향상시켜도 이익이 없습니다." %}

위에서, intensity가 증가함에 따라 (왼쪽에서 오른쪽으로 이동), 처음에는 하드웨어의 임계 arithmetic intensity인 TPU v5e의 경우 240에 도달할 때까지 알고리즘의 성능(FLOP/s)이 선형적으로 증가하는 것을 볼 수 있습니다. 더 낮은 intensity를 가진 알고리즘은 대역폭(BW) 제한을 받고 peak 메모리 대역폭에 의해 제한됩니다(빨간색으로 표시). 오른쪽에 있는 알고리즘은 FLOP을 완전히 활용합니다(녹색으로 표시). 여기서 Algo 1은 통신 제한을 받고 총 하드웨어 FLOP/s의 일부만 사용합니다. Algo 2는 compute-bound입니다. 일반적으로 arithmetic intensity를 증가시키거나 사용 가능한 메모리 대역폭을 증가시켜(BW1에서 BW2로 이동) 알고리즘의 성능을 향상시킬 수 있습니다.

### 행렬 곱셈

곧 우리가 가장 좋아하게 될 알고리즘인 행렬 곱셈(일명 matmul)을 살펴봅시다. $X * Y \rightarrow Z$로 쓰는데, 여기서 $X$는 $\text{bf16}[B, D]$ 모양, $Y$는 $\text{bf16}[D, F]$ 모양, $Z$는 $\text{bf16}[B, F]$ 모양입니다. matmul을 하려면 $2DF + 2BD$ 바이트를 로드하고, $2BDF$ FLOP을 수행하고, $2BF$ 바이트를 다시 써야 합니다.<d-footnote>기술적으로는 $BF \times (2D - 1)$ FLOP을 수행하지만 이는 충분히 가깝습니다. 이는 $BDF$번의 곱셈과 $BF * (D-1)$번의 덧셈에서 나옵니다. 섹션 4에 자세한 내용이 있습니다.</d-footnote> <d-footnote>matmul의 출력이 기술적으로는 float32이지만 일반적으로 HBM에 다시 복사하기 전에 bfloat16으로 캐스팅합니다.</d-footnote> 따라서:

$$\begin{equation}
\text{Intensity}(\text{matmul}) = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}
\end{equation}$$

"batch size" $B$가 $D$와 $F$에 비해 작다고 가정하면 좋은 단순화를 얻을 수 있습니다. 그러면 다음을 얻습니다

$$\begin{equation}
\frac{BDF}{BD + DF + BF} \approxeq \frac{BDF}{DF} = B
\end{equation}$$

$$\begin{equation}
\text{Intensity}(\text{matmul}) > \text{Intensity}(\text{TPU}) \implies B > \frac{1.97e14}{8.20e11} = 240
\end{equation}$$

이는 대부분의 모델에서 로컬 **토큰** batch size $B < 1024$이지만 $D$와 $F > 8000$이므로 Transformer matmul에 대한 합리적인 가정입니다. 따라서 로컬 batch size가 240 토큰보다 클 때 compute-bound가 되며, 이는 매우 간단한 규칙입니다!

<p markdown=1 class="takeaway">**핵심:** bfloat16 matmul이 대부분의 TPU에서 compute-bound가 되려면, 로컬 토큰 batch size가 240보다 커야 합니다.<d-footnote>이는 일반적인 의미의 batch size, 즉 시퀀스의 batch size가 _아니라는_ 점에 유의하세요. 대부분의 rooflines는 토큰들이 같은 시퀀스에 속하든 다른 시퀀스에 속하든 상관없이 순전히 토큰 수에 의존한다는 것이 밝혀졌습니다. 예를 들어 128개의 GPU에서 4096 토큰의 512 시퀀스로 구성된 batch size가 있다면, 총 batch size는 `512 * 4096 = 2M` 토큰이고, 로컬 batch size는 16k 토큰입니다.</d-footnote></p>

이는 아래 문제들에서 살펴볼 몇 가지 주목할 만한 주의사항, 특히 양자화와 관련된 사항들(예: activation을 양자화하지만 여전히 full-precision FLOP을 수행하는 경우)과 함께 제공되지만, 기억할 만한 좋은 규칙입니다. GPU의 경우, 이 숫자는 약간 높지만(300에 가까움), 일반적으로 같은 결론이 성립합니다. [큰 matmul을 작은 matmul들로 분해](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#your-first-matrix-multiplication-kernel)할 때 타일 크기도 중요합니다.<d-footnote>큰 행렬 곱셈을 할 때, VMEM/SMEM/TMEM(더 높은 대역폭의 on-chip 메모리)에 맞는 더 작은 타일로 분해해야 합니다. 이로 인해 청크를 여러 번 로드하게 되므로 더 이상 $O(N^2)$ 바이트만 로드하는 것이 정확하지 않습니다. 타일 크기가 $bm$, $bk$, $bn$인 $(m, k) \cdot (k, n)$ matmul을 고려해보세요. $tm = m / bm$ 등이라고 하면, 총 FLOP은 $2 \cdot tm \cdot tn \cdot tk \cdot bm \cdot bk \cdot bn$이고 총 바이트는 $2 \cdot tm \cdot tn \cdot (tk \cdot (bm \cdot bk + bk \cdot bn) + 2 \cdot bm \cdot bn)$입니다. 마지막 항을 무시하면, intensity는 $bm \cdot bn / (bm + bn)$이 되며, 이는 위와 유사합니다.</d-footnote> [다음 섹션](../tpus)에서 하위 수준의 GPU 및 TPU 세부 사항을 논의할 것입니다.

### 네트워크 통신 rooflines

지금까지 논의한 모든 rooflines는 메모리 대역폭 rooflines였으며, _모두 단일 칩 내에서_였습니다. 이를 규칙으로 받아들여서는 안 됩니다. 실제로, 이 책에서 우리가 관심을 가질 대부분의 rooflines는 칩 간 통신과 관련이 있습니다: 보통 여러 TPU에 걸쳐 샤딩된 행렬과 관련된 행렬 곱셈입니다.

다소 인위적인 예를 들어보면, 2개의 TPU/GPU에 걸쳐 (D 차원을 따라) 균등하게 분할된 두 개의 큰 행렬 $X\sim \text{bfloat16[B, D]}$와 $Y \sim \text{bfloat16[D, F]}$를 곱하고 싶다고 가정해봅시다. 이 곱셈을 하기 위해 ([섹션 3](../sharding)에서 보게 될 것처럼), 각 TPU에서 각 행렬의 절반을 곱할 수 있습니다 (TPU 0에서 `A = X[:, :D // 2] @ Y[:D // 2, :]`, TPU 1에서 `B = X[:, D // 2:] @ Y[D // 2:, :]`) 그 다음 결과인 "partial sum"을 다른 TPU로 복사하고 더합니다. 각 방향으로 `4.5e10` 바이트를 복사할 수 있고 각 칩에서 `1.97e14` FLOP/s를 수행할 수 있다고 가정합시다. $T_\text{math}$와 $T_\text{comms}$는 무엇일까요?

$T_\text{math}$는 각 TPU가 절반의 작업을 하므로 분명히 이전의 절반입니다<d-footnote>두 partial sum을 더하는 데 필요한 FLOP(또 다른 DF 덧셈)을 무시하고 있지만, 이는 기본적으로 무시할 수 있습니다.</d-footnote>:

$$T_\text{math} = \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}} = \frac{BDF}{1.97e14}$$

이제 $T_\text{comms}$는 어떨까요? 이제 이는 칩 간 통신 시간을 의미합니다! 이는 단순히 전송된 총 바이트를 네트워크 대역폭으로 나눈 것입니다:

$$T_\text{comms} = \frac{2BF}{\text{Network Bandwidth}} = \frac{2BF}{4.5e10}$$

따라서 우리는 $$\text{Intensity}(\text{matmul (2-chips)}) > \text{Intensity}(\text{TPU w.r.t. inter-chip network})$$일 때 compute-bound가 되거나, 동등하게 $\frac{BDF}{2BF} = \frac{D}{2} > \frac{1.97e14}{4.5e10} = 4377$ 또는 $D > 8755$일 때 compute-bound가 됩니다. 이전과는 달리, 이제 임계 임계값이 $B$가 아닌 $D$에 의존한다는 점에 주목하세요! 왜 그런지 생각해보세요. 이는 하나의 예일 뿐이지만, 이런 종류의 roofline이 여러 TPU에 걸쳐 연산을 병렬화할 수 있는 시점을 아는 데 중요하다는 점을 강조합니다.

## 몇 가지 연습 문제

**문제 1 [int8 matmul]:** bfloat16 대신 int8 정밀도(매개변수당 1바이트)로 $X[B, D] \cdot_D Y[D, F] \rightarrow Z[B, F]$를 하고 싶다고 가정합시다.<d-footnote>여기서 그리고 전체에 걸쳐 우리는 곱셈이 D 차원에 걸쳐 수축을 수행한다는 것을 나타내기 위해 $A \cdot_D B$ 표기법을 사용할 것입니다. 이는 einsum 표기법의 남용입니다.</d-footnote>

1. 메모리에서 몇 바이트를 로드해야 하나요? 메모리에 몇 바이트를 다시 써야 하나요?
2. 총 몇 개의 OP가 수행되나요?
3. Arithmetic intensity는 무엇인가요?
4. $T_\text{math}$와 $T_\text{comms}$에 대한 roofline 추정은 무엇인가요? 전체 연산의 런타임에 대한 합리적인 상한과 하한은 무엇인가요?

HBM 대역폭이 `8.1e11` bytes/s이고 int8 peak OP/s가 `3.94e14`라고 가정합니다.

{% details 답을 보려면 여기를 클릭하세요. %}

1. 매개변수를 int8로 저장하므로 매개변수당 1바이트를 가지므로, HBM에서 $$BD + DF$$ 바이트를 로드하고 $$BF$$를 다시 씁니다.
2. 이는 bfloat16에서와 동일하지만, 이론적으로 int8 OP/s가 더 빨라야 합니다. 따라서 여전히 $2BDF$ FLOP입니다.
3. Arithmetic intensity는 $$2BDF / (BD + DF + BF)$$입니다. $$B \ll D$$와 $$B \ll F$$에 대해 위와 같은 가정을 하면, arithmetic intensity는 $$2B$$가 되며, 이는 우리의 규칙이 $B > \text{HBM int8 arithmetic intensity} / 2$가 됨을 의미합니다. 주어진 숫자를 사용하면, 이 int8 intensity는 `3.94e14 / 8.1e11 = 486`이므로, 규칙은 $B > 486 / 2 = 243$입니다. 이는 기본적으로 변하지 않았다는 점에 주목하세요!
4. $$T_\text{math} = 2BDF / 3.94e14$$이고 $$T_\text{comms} = (BD + DF + BF) / 8.1e11$$이므로, 합리적인 하한은 $$\max(T_\text{math}, T_\text{comms})$$이고 상한은 $$T_\text{math} + T_\text{comms}$$입니다.

{% enddetails %}

**문제 2 [int8 + bf16 matmul]:** 실제로는 종종 다른 가중치 대 activation 양자화를 하므로, 가중치를 매우 낮은 정밀도로 저장하지만 activation(과 계산)을 더 높은 정밀도로 유지할 수 있습니다. 가중치를 int8로 양자화하지만 activation(과 계산)을 bfloat16으로 유지하고 싶다고 가정합시다. 어떤 batch size에서 compute bound가 될까요? `1.97e14` bfloat16 FLOP/s라고 가정합니다.

*힌트: 이는 구체적으로 `bfloat16[B, D] * int8[D, F] -> bfloat16[B, F]`를 의미하며, 여기서 $B$는 "batch size"입니다.*

{% details 답을 보려면 여기를 클릭하세요. %}

다시 B가 작다고 가정하면, 2BDF bfloat16 FLOP을 가지지만 DF 가중치만 가집니다(bfloat16에서 2DF 대신). 이는 $$2B > 240$$ 또는 $$B > 120$$일 때 compute-bound가 됨을 의미합니다. 이는 훨씬 낮으며, int8 가중치 양자화(상당히 쉽게 할 수 있음)를 할 수 있지만 여전히 bfloat16 FLOP을 수행한다면 효율성에서 의미 있는 향상을 얻는다는 것을 의미합니다(int8 OP이 더 좋을 것이지만).

{% enddetails %}

**문제 3:** 문제 2의 설정을 사용하여, $F = D = 4096$과 $F = D = 1024$에 대해 peak FLOP 대 $B$의 roofline plot을 만드세요. *근사치가 아닌 정확한 로드된 바이트 수를 사용하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

다음은 해당 plot입니다:

{% include figure.liquid path="assets/img/roofline-plot-q3.png" class="img-fluid img-small" %}

두 모델 모두 결국 peak 하드웨어 FLOP/s를 달성하지만, 더 큰 D/F가 더 빨리 달성한다는 점에 주목하세요. D=F=1024는 임계 batch size를 거의 두 배로 만듭니다. 이 그림을 생성하는 코드는 다음과 같습니다:

```py
import matplotlib.pyplot as plt
import numpy as np

bs = np.arange(1, 512)

def roofline(B, D, F):
  total_flops = 2*B*D*F
  flops_time = total_flops / 1.97e14
  comms_time = (2*B*D + D*F + 2*B*F) / 8.2e11
  total_time = np.maximum(flops_time, comms_time)
  return total_flops / total_time

roofline_big = roofline(bs, 4096, 4096)
roofline_small = roofline(bs, 1024, 1024)

plt.figure(figsize=(8, 4))
plt.plot(bs, roofline_big, label='F=D=4096')
plt.plot(bs, roofline_small, label='F=D=1024')
plt.legend()
plt.xlabel('batch size')
plt.ylabel('peak bfloat16 FLOPs/s on TPU v5e')
plt.grid()
```

{% enddetails %}

**문제 4:** 각 batch 요소에 대해 다른 행렬을 가진다고 상상하며 $\text{int8[B, D]} *_D \text{int8[B, D, F]} \rightarrow \text{int8[B, F]}$를 수행하고 싶다면 어떨까요? 이 연산의 arithmetic intensity는 무엇인가요?

{% details 답을 보려면 여기를 클릭하세요. %}

총 FLOP과 통신을 먼저 살펴봅시다.

1. 총 FLOP: 같은 수의 $$BD \times DF$$ matmul을 하고 있으므로 FLOP은 기본적으로 동일합니다(이는 섹션 4에서 더 자세히 논의됩니다). 따라서 이는 단순히 $$2BDF$$입니다.
2. 총 통신: 여기서 훨씬 더 많은 통신이 있습니다: $$BD + BDF + BF$$.
3. 따라서, 우리의 arithmetic intensity는 이제 실제로 $$2BDF / (BD + BDF + BF)$$입니다. $$BDF$$가 분모를 지배하므로, 이는 대략 $$2$$입니다. batch size에 의존하는 대신, 이는 본질적으로 상수입니다. 이는 무엇을 하든 기본적으로 항상 통신 제한을 받을 것이므로 나쁩니다.

{% enddetails %}

**문제 5 [GPU용 메모리 Rooflines]:** [NVIDIA가 H100에 대해 제공한 스펙 시트](https://www.nvidia.com/en-us/data-center/h100/)를 사용하여, 행렬 곱셈이 compute-bound가 되는 batch size를 계산하세요. *Tensor Core FLOP 수치는 구조화된 희소성으로만 달성 가능하므로 실제 값의 두 배라는 점에 유의하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

스펙 시트에서, 보고된 bfloat16 FLOP 값이 "희소성 포함"이라는 별표와 함께 `1.979e15` FLOP/s라는 것을 볼 수 있습니다. 희소성 없이는 실제 값이 이것의 절반이므로, `1e15` FLOP/s에 가깝습니다. 메모리 대역폭은 3.35TB/s, 즉 `3.35e12` bytes/second입니다. 따라서 $B_\text{crit}$는 `1e15 / 3.35e12 = 298`이며, 이는 TPU와 상당히 유사합니다.

{% enddetails %}

<h3 markdown=1 class="next-section">Part 1은 여기까지입니다! 실제 TPU가 FLOP과 통신을 어떻게 처리하는지 살펴보는 Part 2는 [여기를 클릭하세요](../tpus).</h3>