---
layout: distill
title: "TPU에 대해 생각하는 방법"
# permalink: /main/
description: "이 섹션은 TPU가 어떻게 작동하는지, multi-chip 훈련과 추론을 가능하게 하기 위해 어떻게 네트워크로 연결되는지, 그리고 이것이 우리가 좋아하는 알고리즘의 성능에 어떤 영향을 미치는지에 대한 모든 것입니다. GPU 사용자들에게도 좋은 내용이 있습니다!"
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting

section_number: 2

previous_section_url: "../roofline"
previous_section_name: "Part 1: Rooflines"

next_section_url: ../sharding
next_section_name: "Part 3: Sharding"

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
  - name: TPU란 무엇인가?
  - name: TPU Networking
  - name: 핵심 내용
  - name: 연습 문제
  - name: 부록
  - subsections:
    - name: "부록 A: GPU에 대한 모든 것"
    - name: "부록 B: Systolic array는 어떻게 작동하는가?"
    - name: "부록 C: TPU 내부에 대한 자세한 내용"

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

## TPU란 무엇인가?

**TPU는 기본적으로 행렬 곱셈을 전문으로 하는 compute core(TensorCore라고 함)가 빠른 메모리 스택(high-bandwidth memory 또는 HBM이라고 함)에 연결된 것입니다<d-cite key="tpu_paper"></d-cite>.** 다음은 다이어그램입니다:

{% include figure.liquid path="assets/img/tpu-chip.png" class="img-fluid" caption="<b>그림:</b> TPU 칩의 기본 구성 요소. TensorCore는 회색 왼쪽 박스로, matrix-multiply unit (MXU), vector unit (VPU), 그리고 vector memory (VMEM)를 포함합니다." %}

TensorCore를 기본적으로 정말 좋은 행렬 곱셈 기계라고 생각할 수 있지만, 주목할 만한 몇 가지 다른 기능이 있습니다. TensorCore에는 세 가지 핵심 유닛이 있습니다:

* **MXU** (Matrix Multiply Unit)는 TensorCore의 핵심입니다. 대부분의 TPU 세대에서, 이것은 systolic array를 사용하여 8 사이클마다 하나의 `bfloat16[8,128] @ bf16[128,128] -> f32[8,128]` 행렬 곱셈을 수행합니다<d-footnote>TPU v6e (Trillium)는 256x256 MXU를 가지고 있는 반면, 이전의 모든 세대는 128x128을 사용합니다</d-footnote> (자세한 내용은 <a href="#부록-b-systolic-array는-어떻게-작동하는가">부록 B</a> 참조).
  * 이는 TPU v5e에서 1.5GHz에서 MXU당 약 `5e13` bf16 FLOP/s입니다. 대부분의 TensorCore는 2개 또는 4개의 MXU를 가지므로, 예를 들어 TPU v5e의 총 bf16 FLOP/s는 `2e14`입니다.
  * TPU는 또한 더 높은 처리량을 가진 더 낮은 정밀도 matmul을 지원합니다 (예: 각 TPU v5e 칩은 `4e14` int8 OP/s를 수행할 수 있습니다).

* **VPU** (Vector Processing Unit)는 ReLU 활성화나 벡터 간의 pointwise 덧셈 또는 곱셈과 같은 일반적인 수학 연산을 수행합니다. Reduction(합계)도 여기서 수행됩니다. <a href="#부록-c-tpu-내부에-대한-자세한-내용">부록 C</a>에서 자세한 내용을 제공합니다.
* **VMEM** (Vector Memory)은 compute unit에 가까운 TensorCore에 위치한 on-chip scratchpad입니다. HBM보다 훨씬 작지만(예: TPU v5e에서 128 MiB) MXU에 대한 대역폭이 훨씬 높습니다. VMEM은 CPU의 L1/L2 캐시와 어느 정도 비슷하게 작동하지만 훨씬 크고 프로그래머가 제어합니다. HBM의 데이터는 TensorCore가 어떤 계산이든 수행하기 전에 VMEM으로 복사되어야 합니다.

**TPU는 행렬 곱셈에서 매우, 매우 빠릅니다**. 주로 하는 일이고 잘합니다. 현재까지 가장 강력한 TPU 중 하나인 [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 core당 `2.5e14` bf16 FLOP/second 또는 칩당 `5e14` bf16 FLOP/sec를 수행할 수 있습니다. 8960개 칩의 단일 pod는 초당 4 exaflop을 수행할 수 있습니다. 그것은 *매우 많습니다*. 세계에서 가장 강력한 슈퍼컴퓨터 중 하나입니다. 그리고 Google은 많이 가지고 있습니다.<d-footnote>TPU, 특히 그들의 systolic array는 행렬 곱셈이 $O(n^2)$ 바이트에 대해 $O(n^3)$ 계산을 사용하는 몇 안 되는 알고리즘 중 하나이기 때문에 강력한 하드웨어 가속기입니다. 이것은 일반적인 ALU가 메모리 대역폭이 아닌 계산에 의해 병목되기 매우 쉽게 만듭니다.</d-footnote>

위의 다이어그램에는 제어 흐름 처리에 사용되고 <a href="#부록-c-tpu-내부에-대한-자세한-내용">부록 C</a>에서 간략히 논의되는 SMEM과 scalar unit과 같은 몇 가지 다른 구성 요소도 포함되어 있지만, 이해하는 데 중요하지는 않습니다. 반면에 HBM은 중요하고 상당히 간단합니다:

* **HBM** (High Bandwidth Memory)은 TensorCore에서 사용할 tensor를 저장하는 빠른 메모리의 큰 덩어리입니다. HBM은 일반적으로 수십 기가바이트 규모의 용량을 가집니다 (예: [TPU v5e는 16GiB의 HBM을 가집니다](https://cloud.google.com/tpu/docs/v5e#system_architecture)).

  * 계산에 필요할 때, tensor는 VMEM을 통해 (아래 참조) HBM에서 MXU로 스트리밍되고 결과는 VMEM에서 HBM으로 다시 쓰여집니다.

  * HBM과 TensorCore 사이의 대역폭(VMEM을 통해)은 "HBM 대역폭"(보통 1-2TB/sec 정도)으로 알려져 있으며 메모리 제한 워크로드에서 계산이 얼마나 빠르게 수행될 수 있는지를 제한합니다.

**일반적으로, 모든 TPU 연산은 파이프라인되고 겹쳐집니다.** matmul $X \cdot A \to Y$를 수행하기 위해, TPU는 먼저 행렬 $A$와 $X$의 청크를 HBM에서 VMEM으로 복사한 다음, ($X$에 대해) 8x128과 ($A$에 대해) 128x128의 청크를 곱하는 MXU에 로드한 다음, 결과를 청크별로 HBM에 다시 복사해야 합니다. 이를 효율적으로 하기 위해, matmul은 VMEM으로의/에서의 복사가 MXU 작업과 겹치도록 파이프라인됩니다. 이를 통해 MXU가 메모리 전송을 기다리는 대신 계속 작업할 수 있어, matmul을 메모리 제한이 아닌 계산 제한으로 유지합니다.

다음은 HBM에서 elementwise product를 수행하는 방법의 예입니다:

{% include figure.liquid path="assets/img/pointwise-product.gif" caption="<b>그림:</b> HBM에서 로드된 바이트로 TPU에서 수행되는 pointwise product를 보여주는 애니메이션. 바이트가 청크로 메모리에서 스트리밍되고 전체 배열이 구체화되기를 기다리지 않고 부분 결과가 파이프라인으로 다시 전송되는 방법에 주목하세요." %}

matmul은 VPU/Vector unit 대신 MXU에 로드하고, 같은 가중치 청크가 여러 activation 청크에 사용되므로 로드와 저장이 다른 순서로 발생한다는 점을 제외하면 거의 동일하게 보일 것입니다. 데이터 청크가 VMEM으로, 그 다음 VREG(vector register)로, 그 다음 Vector Unit으로, 그 다음 다시 VMEM과 HBM으로 스트리밍되는 것을 볼 수 있습니다. 곧 보게 될 것처럼, HBM에서 VMEM으로의 로드가 Vector Unit(또는 MXU)의 FLOP보다 느리면, VPU나 MXU의 작업을 굶주리게 하므로 "대역폭 제한"이 됩니다.

<p markdown=1 class="takeaway">**핵심:** TPU는 매우 간단합니다. HBM에서 VMEM으로 가중치를 로드한 다음, VMEM에서 초당 약 200조 번의 곱셈-덧셈을 수행할 수 있는 systolic array로 로드합니다. HBM $\leftrightarrow$ VMEM과 VMEM $\leftrightarrow$ systolic array 대역폭은 TPU가 효율적으로 수행할 수 있는 계산에 대한 근본적인 한계를 설정합니다.</p>

**VMEM과 arithmetic intensity:** VMEM은 HBM보다 훨씬 작지만 MXU에 대한 대역폭이 훨씬 높습니다. [섹션 1](../roofline)에서 본 것처럼, 이는 알고리즘이 모든 입력/출력을 VMEM에 맞출 수 있다면, 통신 병목에 걸릴 가능성이 훨씬 낮다는 것을 의미합니다. 이는 낮은 arithmetic intensity를 가진 계산에 특히 도움이 됩니다: VMEM 대역폭은 HBM 대역폭보다 약 22배 높으므로 VMEM에서 읽기/쓰기하는 MXU 연산은 peak FLOP 활용을 달성하기 위해 10-20의 arithmetic intensity만 필요합니다. 즉, 가중치를 HBM 대신 VMEM에 맞출 수 있다면, 훨씬 작은 batch size에서 행렬 곱셈이 FLOP 제한이 될 수 있습니다. 그리고 근본적으로 더 낮은 arithmetic intensity를 가진 알고리즘도 여전히 효율적일 수 있다는 것을 의미합니다. VMEM이 너무 작아서 이것이 종종 도전입니다.<d-footnote>우리는 때때로 matmul의 로딩 비용을 가리기 위해 VMEM에 가중치를 미리 로드하는 것을 의미하는 VMEM prefetching에 대해 이야기합니다. 예를 들어, 일반적인 Transformer에서는 때때로 attention 중에 큰 feed-forward 가중치를 VMEM에 로드할 수 있으며, 이는 메모리 대역폭 제한이라면 가중치 로드의 비용을 숨길 수 있습니다. 이는 가중치가 충분히 작거나 VMEM에 여유 공간과 함께 단일 레이어를 맞출 수 있을 정도로 샤딩되어야 합니다.</d-footnote>

{% include figure.liquid path="assets/img/tpu-bandwidth.png" class="img-fluid" %}

**TPU 칩은 일반적으로 (항상은 아니지만) 메모리를 공유하고 두 배의 FLOP을 가진 하나의 큰 accelerator로 생각할 수 있는 두 개의 TPU core로 구성됩니다** ("megacore" 구성으로 알려짐). 이는 TPU v4 이후로 사실입니다. 더 오래된 TPU 칩은 별도의 메모리를 가지고 있으며 두 개의 별도 accelerator로 간주됩니다 (TPU v3 이하). TPU v5e와 같은 추론 최적화 칩은 칩당 하나의 TPU core만 가집니다.

{% include figure.liquid path="assets/img/cores.png" class="img-fluid img-small" %}

**칩들**은 **PCIe 네트워크를 통해 CPU 호스트에 연결된 'tray'에 4개씩 배치됩니다.** 이는 대부분의 독자들이 친숙할 형식으로, Colab이나 단일 TPU-VM을 통해 노출되는 4개의 칩(8개의 core, 하지만 보통 4개의 논리적 megacore로 취급됩니다)입니다. TPU v5e와 같은 추론 칩의 경우, 1개 대신 호스트당 2개의 tray를 가지지만, 칩당 1개의 core만 가져서 8개의 칩 = 8개의 core를 제공합니다.<d-footnote>Cloud TPU VM에서, 각 tray는 별도의 VM의 일부로 노출되므로, 다시 한 번 4개의 core가 보입니다.</d-footnote>

{% include figure.liquid path="assets/img/pcie.png" class="img-fluid" %}

**PCIe 대역폭은 제한적입니다:** HBM $\leftrightarrow$ VMEM 링크와 마찬가지로, CPU $\leftrightarrow$ HBM PCIe 연결은 호스트 메모리에서 HBM으로 또는 그 반대로 얼마나 빠르게 로드할 수 있는지를 제한하는 특정 대역폭을 가집니다. 예를 들어, TPU v4의 PCIe 대역폭은 각 방향으로 16GB/second이므로, HBM보다 거의 100배 느립니다. 호스트(CPU) RAM에 데이터를 로드/오프로드할 *수는* 있지만, 매우 빠르지는 않습니다.

## TPU Networking

**칩들은 Pod에서 ICI 네트워크를 통해 서로 연결됩니다**. 이전 세대(TPU v2와 TPU v3), 추론 칩(예: TPU v5e), 그리고 Trillium(TPU v6e)에서, ICI("inter-chip interconnects")는 가장 가까운 4개의 이웃을 연결합니다(2D torus를 형성하기 위한 edge 링크와 함께). TPU v4와 TPU v5p는 가장 가까운 6개의 이웃에 연결됩니다(3D torus 형성). 이러한 연결은 호스트를 **통하지 않고**, 칩 간의 직접 링크라는 점에 주목하세요.

{% include figure.liquid path="assets/img/ici-wraparound.png" class="img-fluid img-small" %}

토로이달 구조는 임의의 두 노드 사이의 최대 거리를 $N$에서 $N / 2$로 줄여 통신을 훨씬 빠르게 만듭니다. TPU는 또한 노드 간의 평균 거리를 더욱 줄이기 위해 torus를 뫼비우스 띠와 같은 토폴로지로 감싸는 "twisted torus" 구성을 가집니다.

**TPU pod(ICI로 연결됨)는 정말 커질 수 있습니다:** 최대 pod 크기(**superpod**라고 함)는 TPU v4의 경우 `16x16x16`이고 TPU v5p의 경우 `16x20x28`입니다. 이러한 큰 pod는 재구성 가능한 `4x4x4` 칩의 큐브로 구성되며 [optical wraparound 링크](https://arxiv.org/pdf/2208.10041)<d-footnote>optical switch는 단순히 같은 ICI 대역폭을 가진 재구성 가능한 연결입니다. 단지 우리가 wraparound 링크를 유지하면서 큐브를 연결할 수 있게 해줍니다.</d-footnote>로 연결되어 매우 큰 토폴로지를 연결하도록 재구성할 수 있습니다.

{% include figure.liquid path="assets/img/tpu-rack.png" class="img-fluid" %}

더 작은 토폴로지(예: `2x2x1`, `2x2x2`)도 요청할 수 있지만, wraparound는 없습니다. 이는 대부분의 통신 시간을 일반적으로 두 배로 만들기 때문에 중요한 주의사항입니다. 전체 큐브의 배수(예: `4x4x4` 또는 `4x4x8`)는 optical switch에서 제공하는 wraparound를 가질 것입니다.<d-footnote>`2x2x4`는 전체 큐브에서만 사용 가능한 optical switch에서 제공되므로 wraparound가 없다는 점에 주목하세요. 하지만 TPU v5e 8x16은 재구성 가능한 optical 네트워킹을 사용하지 않으므로 긴 축에서 wraparound를 _가질_ 것입니다.</d-footnote>

{% include figure.liquid path="assets/img/subslices.png" class="img-fluid" %}

TPU v5e와 Trillium pod는 크기 16의 축을 따라 wraparound를 가진 단일 `16x16` 2D torus로 구성됩니다 (`8x16`은 긴 축에서 wraparound를 가진다는 의미). TPU v5e와 v6e(Trillium)는 16x16 torus를 넘어 확장할 수 없지만 pod는 여전히 TPU 호스트를 서로 연결하는 표준 데이터센터 네트워킹(DCN)을 통해 서로 통신할 수 있습니다. 다시, $<16$ 차원에서 wrap 없이 더 작은 토폴로지를 요청할 수 있습니다.

{% include figure.liquid path="assets/img/more-subslices.png" class="img-fluid" %}

**이러한 nearest-neighbor 연결성은 TPU와 GPU 사이의 핵심 차이점입니다**. GPU는 TPU와 같은 local 연결을 사용하기보다는 모든 GPU 간의 point-to-point 연결을 근사하는 스위치 계층으로 연결됩니다. 일반적으로, 노드 내의 GPU(H100의 경우 8개 GPU 또는 B200의 경우 최대 500개)는 직접 연결되는 반면, 더 큰 토폴로지는 각 GPU 간에 O(log(N)) hop이 필요합니다. 한편으로는, 이것이 GPU가 노드 내에서 단일 저지연 hop으로 임의의 데이터를 보낼 수 있다는 것을 의미합니다. 다른 한편으로는, TPU가 극적으로 더 저렴하고(NVLink 스위치가 비싸므로) 연결하기 더 간단하며, 디바이스당 링크 수와 디바이스당 대역폭이 일정하므로 훨씬 더 큰 토폴로지로 확장할 수 있습니다.

**ICI는 DCN에 비해 매우 빠르지만, 여전히 HBM 대역폭보다는 느립니다.** 예를 들어, [TPU v5p](https://cloud.google.com/tpu/docs/v5p#system_architecture)는 다음을 가집니다:

* 칩당 `2.5e12` bytes/s (2.5 TB/s)의 HBM 대역폭.
* 칩당 3개의 축과 함께 축당 `9e10` bytes/s (90<d-footnote>위의 페이지는 100 GB/s의 대역폭을 나열하며, 이는 여기에 나열된 것과 약간 다릅니다. TPU ICI 링크는 수행되는 연산에 따라 약간 다른 대역폭을 가집니다. 일반적으로 이 문서의 숫자를 걱정 없이 사용할 수 있습니다.</d-footnote> GB/s)의 ICI 대역폭.
* 호스트당 `2.5e10` bytes/s (25 GB/s)의 DCN (egress) 대역폭. 일반적으로 호스트당 8개의 TPU를 가지므로, 이는 실제로 칩당 `3.1e9` bytes/s에 가깝습니다.

이는 모델을 여러 칩에 걸쳐 분할할 때, 느린 cross-device 통신으로 MXU를 병목시키지 않도록 주의해야 함을 의미합니다.

**Multi-slice 훈련:** ICI로 연결된 TPU의 집합을 **slice**라고 합니다. 다른 slice는 DCN을 사용하여 서로 연결될 수 있으며, 예를 들어 다른 pod의 slice를 연결할 수 있습니다. DCN이 ICI보다 훨씬 느린 연결이므로, 계산이 DCN에서 오는 데이터를 기다려야 하는 정도를 제한하려고 노력해야 합니다. DCN은 호스트 간이므로, DCN을 통해 TPU에서 TPU로 버퍼를 전송하려면, 먼저 PCIe를 통해 호스트로 전송한 다음, 네트워크를 통해 egress하고, 대상 호스트 네트워크를 통해 ingress한 다음, PCIe를 통해 HBM으로 전송해야 합니다.

## 핵심 내용

* TPU는 간단하며 대부분의 경우 메모리(매우 빠름), ICI를 통한 다른 칩(상당히 빠름), 그리고 DCN을 통한 데이터센터의 나머지 부분(어느 정도 빠름)에 연결된 행렬 곱셈 유닛으로 생각할 수 있습니다.

* 통신은 속도 순서로 다양한 네트워크 대역폭에 의해 제한됩니다:
  * HBM 대역폭: TensorCore와 관련 HBM 사이.
  * ICI 대역폭: TPU 칩과 가장 가까운 4개 또는 6개의 이웃 사이.
  * PCIe 대역폭: CPU 호스트와 관련 칩의 tray(들) 사이.
  * DCN 대역폭: 여러 CPU 호스트 사이, 일반적으로 ICI로 연결되지 않은 호스트들.

* **slice 내에서, TPU는 ICI를 통해 가장 가까운 이웃에만 연결됩니다.** 이는 slice의 먼 칩 간의 ICI를 통한 통신이 먼저 중간 칩을 hop해야 함을 의미합니다.

* **가중치 행렬은 MXU를 채우기 위해 양쪽 차원에서 최소 크기 128**(TPU v6에서는 256)로 패딩되어야 합니다 (실제로, 더 작은 축은 128로 패딩됩니다).

* **더 낮은 정밀도 행렬 곱셈이 더 빠른 경향이 있습니다.** TPU는 이를 지원하는 세대의 경우 bfloat16 FLOP보다 대략 2배/4배 빠르게 int8 또는 int4 FLOP을 수행할 수 있습니다. VPU 연산은 여전히 fp32로 수행됩니다.

* TPU compute unit의 병목을 피하기 위해, **각 채널을 통한 통신량이 그 속도에 비례하는지 확인해야 합니다**.

* **우리 칩에 대한 구체적인 숫자들입니다:**

| Model                                      | Pod size | Host size | HBM capacity/chip | HBM BW/chip (bytes/s) | FLOPs/s/chip (bf16) | FLOPs/s/chip (int8) |
| :----------------------------------------- | :------: | :-------: | :---------------: | :-------------------: | :-----------------: | :-----------------: |
| <span class="nowrap-header">TPU v3</span>  |  32x32   |    4x2    |       32GB        |        9.0e11         |       1.4e14        |       1.4e14        |
| <span class="nowrap-header">TPU v4p</span> | 16x16x16 |   2x2x1   |       32GB        |        1.2e12         |       2.75e14       |       2.75e14       |
| <span class="nowrap-header">TPU v5p</span> | 16x20x28 |   2x2x1   |       96GB        |        2.8e12         |       4.59e14       |       9.18e14       |
| <span class="nowrap-header">TPU v5e</span> |  16x16   |    4x2    |       16GB        |        8.1e11         |       1.97e14       |       3.94e14       |
| <span class="nowrap-header">TPU v6e</span> |  16x16   |    4x2    |       32GB        |        1.6e12         |       9.20e14       |       1.84e15       |

Host size는 단일 호스트에 연결된 TPU의 토폴로지를 의미합니다 (예: TPU v5e는 4x2 토폴로지의 8개 TPU에 연결된 단일 CPU 호스트를 가집니다). 다음은 interconnect 수치입니다:

| Model       | ICI BW/link (one-way, bytes/s) | ICI BW/link (bidi, bytes/s) |
| :---------- | :----------------------------: | :-------------------------: |
| **TPU v3**  |              1e11              |            2e11             |
| **TPU v4p** |             4.5e10             |            9e10             |
| **TPU v5p** |              9e10              |           1.8e11            |
| **TPU v5e** |             4.5e10             |            9e10             |
| **TPU v6e** |              9e10              |           1.8e11            |

단방향(unidirectional) 대역폭은 하드웨어에 더 가깝지만 양방향(bidirectional) 대역폭은 전체 링을 포함하는 방정식에서 더 자주 발생하므로 둘 다 포함합니다.<d-footnote>bidi(bidirectional) 대역폭이란 단일 링크를 따라 양방향으로 보낼 수 있는 총 바이트, 또는 동등하게 양방향 링크를 효율적으로 사용할 수 있다고 가정할 때 특정 축을 따라 단일 TPU에서 나가는 총 바이트 수를 의미합니다. 이는 작동하는 링, 즉 특정 축에서 wraparound 연결이 있을 때 참입니다. 이는 전체 16 축을 가질 때 추론 칩에서 발생하거나, 4의 배수인 축을 가질 때 훈련 칩(v*p)에서 발생합니다. 양방향 통신을 포함하는 계산에서 자주 나타나므로 양방향 대역폭을 사용하는 것을 선호합니다.</d-footnote>

PCIe 대역폭은 일반적으로 칩당 약 `1.5e10` bytes/second<d-footnote>Trillium (TPU v6e)는 32GB/s로, v5보다 약 2배 높습니다.</d-footnote>이고, DCN 대역폭은 일반적으로 호스트당 약 `2.5e10` bytes/second입니다. 완전성을 위해 단방향과 양방향 대역폭을 모두 포함합니다. 일반적으로 전체 wraparound 링에 접근할 수 있을 때는 양방향 대역폭이 더 유용한 숫자이고, 단방향 대역폭이 하드웨어에 더 가깝습니다.

## 연습 문제

이러한 숫자들은 약간 건조하지만, 모델 성능에 대한 기본적인 roofline 추정을 할 수 있게 해줍니다. 이것이 왜 유용한지 설명하기 위해 몇 가지 문제를 해결해봅시다. Part 3에서 더 많은 예제를 볼 것입니다.

**문제 1 [LLM 지연 시간 제한]:** 32개의 TPU v4p에 분할된 bf16의 200B 매개변수 모델에서 샘플링하고 싶다고 가정합시다. HBM에서 systolic array로 모든 매개변수를 로드하는 데 얼마나 걸릴까요? *힌트: 위의 숫자들을 사용하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 우리는 32개의 칩에서 `sizeof(bf16) * 200e9 = 400e9` 바이트를 로드하고 있으며, 이는 칩당 12.5e9 바이트를 의미하고, 각각 1.23e12의 HBM 대역폭을 가지므로 로드에 약 10ms가 걸립니다.

이는 꽤 멋진데, *모델에서 샘플링하는 지연 시간의 합리적인 하한*이기 때문입니다. 각 샘플링 단계는 HBM에서 모든 매개변수를 로드해야 하므로, 10ms보다 적게 걸릴 수 없습니다. 실제로는, 작은 batch size에서 이는 달성 가능에 가깝습니다.

{% enddetails %}

**문제 2 [TPU 세부사항]:** 전체 TPU v5e pod를 고려해보세요. 총 CPU 호스트는 몇 개입니까? TPU TensorCore는 몇 개입니까? 전체 pod의 총 FLOP/s는 얼마입니까? 총 HBM은 얼마입니까? TPU v5p pod에 대해서도 같은 연습을 해보세요.

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** TPU v5e의 경우, 각 pod는 `16x16`이고 각 호스트는 4x2 slice이므로, `16*16 / 8 = 32`개의 호스트를 가집니다. TPU v5e의 경우, 각 TPU는 하나의 core만 가지므로, 256개의 TensorCore를 가집니다. 총 FLOP/s는 bfloat16에서 `16*16*2e14 = 5.1e16`입니다. 각 칩은 16GB의 HBM을 가지므로, `256 * 16 = 4TB`의 메모리입니다.

전체 TPU v5p pod의 경우, `16x20x28`개의 칩을 가지고 각 호스트는 2x2x1이므로, `16*20*28 / 2*2 = 2,240`개의 호스트를 가집니다. TPU v5p의 경우, 각 TPU는 두 개의 TensorCore를 가지므로, `8960 * 2 = 17,920`개의 core를 가집니다. 총 FLOP/s는 bfloat16에서 `8960 * 4.5e14 = 4e18`입니다. 각 칩은 96GB의 HBM을 가지므로, `8960 * 96 = 860TB`의 메모리입니다.

{% enddetails %}

**문제 3 [PCIe operational intensity]:** 큰 가중치 행렬 $A$를 $\text{bfloat16}[D, F]$ 타입으로, 그리고 activation의 batch $x$를 $\text{bfloat16}[B, D]$ 타입으로 호스트 DRAM에 저장하고 이들에 대해 행렬 곱셈을 하도록 강제되었다고 상상해보세요. 이는 단일 호스트에서 실행되고 있으며, 여기에 연결된 단일 TPU v6e 칩을 사용하고 있습니다. $B \ll D$이고 $F = 4D$라고 가정할 수 있습니다 (이것들이 합리적인 가정인 이유를 향후 장에서 볼 것입니다). PCIe에 대해 FLOP 제한을 유지하는 데 필요한 가장 작은 batch size $B$는 무엇입니까? PCIe 대역폭을 1.5e10 bytes/second라고 가정하세요.

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 우리는 $2BDF$ 부동 소수점 연산을 수행해야 하고, 각 칩은 초당 `9.2e14` 부동 소수점 연산을 수행할 수 있습니다. 그러면 수행하는 데 $2BDF / 9.2e14$ 초가 필요합니다. DRAM에서 $2DF + 2BD$ 바이트를 로드하고, $2BF$ 바이트를 다시 써야 합니다. PCIe 전송 속도에 의해 병목되므로, TPU로 또는 TPU에서 데이터를 전송하는 데 $2 \cdot (BD + DF + BF) / 1.5e10$ 초가 필요합니다. 계산이 가중치 로딩보다 오래 걸리기를 원하므로, 모든 가중치 로딩을 계산과 겹칠 수 있다고 가정하면, $2BDF / 9.2e14 > 2 \cdot (BD + DF + BF) / 1.5e10$을 원합니다. $B \ll D$이고 $F = 4D$라는 가정을 사용하여 이를 단순화하면 다음을 얻습니다

$$\frac{8BD^2}{9.2e14} > \frac{8D^2}{1.5e10}$$

또는

$$B > \frac{9.2e14}{1.5e10} \simeq 61,000$$

{% enddetails %}

**문제 4 [일반 matmul 지연 시간]:** 가중치 행렬 int8[16384, 4096]에 크기가 int8[B, 4096]인 activation 행렬을 곱하고 싶다고 가정해봅시다. 여기서 B는 알 수 없는 batch size입니다. 시작하기 위해 1개의 TPUv5e에 있다고 가정합시다.

1. 이 곱셈은 B의 함수로 얼마나 걸릴까요? *힌트: HBM에서 배열을 로드하는 데 걸리는 시간과 곱셈이 실제로 걸리는 시간을 계산하는 것이 도움이 될 수 있습니다. 무엇이 병목이 되고 있나요?*
2. 이 연산을 VMEM에서 실행하고 싶다면 어떨까요? B의 함수로 얼마나 걸릴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** (1) 수행해야 하는 부동 소수점 연산의 수는 $2 \cdot 4096 \cdot 16384 \cdot B = 1.3e8 \cdot B$입니다. 따라서 $T_{\text{math}} = (1.3e8 \cdot B) / 3.94e14$ 초입니다. HBM에서 VMEM으로 $16384 \cdot 4096 + 4096 \cdot B$ 바이트를 로드하고, VMEM에서 HBM으로 $16384 \cdot B$ 바이트를 다시 써야 합니다. 이는 $T_{\text{comms}} = (6.7e7 + 2e4\cdot B) / 8.1e11$ 초를 의미합니다. 통신과 계산의 최대한 많은 겹침을 가정하면, 전체 곱셈은 대략 다음 시간이 걸립니다

$$\max\{T_{\text{math}}, T_{\text{comms}}\} = \max\left\{\frac{6.7e7 + 2e4\cdot B}{8.1e11}, \frac{1.3e8 \cdot B}{3.94e14}\right\}$$

$\frac{6.7e7 + 2e4\cdot B}{8.1e11} < \frac{1.3e8 \cdot B}{3.94e14}$일 때, 즉 $B > 271$일 때 FLOP 제한이 됩니다. 이는 $D$와 $F$의 전체 영향을 고려하기 때문에 아래에서 도출하는 240이란 숫자보다 약간 큽니다.

(2) 대신 VMEM에서 로딩한다면, MXU에 대한 VMEM 대역폭을 HBM $\leftrightarrow$ VMEM 대역폭의 22배로 고려합시다. 이는 데이터 로딩 분모를 8.1e11에서 1.78e13으로 바꾸고, $B > 11$을 얻습니다. 실제로는 모든 VMEM 대역폭을 $W$ 로딩에 전념할 수 없으므로, 실제로는 20에 가까울 것입니다.

{% enddetails %}

**문제 5 [ICI 대역폭]:** TPU v5e `4x4` slice가 있다고 가정합시다. `bfloat16[8, 128, 8192]` 타입의 배열을 `TPU{0,0}`에서 `TPU{3, 3}`로 보내고 싶다고 가정합시다. TPU v5e의 hop당 지연 시간이 $1\mu s$라고 가정합시다.

1. 첫 번째 바이트가 목적지에 언제 도착할까요?
2. 전체 전송은 얼마나 걸릴까요?

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** TPUv5e에서는 2D 연결성을 가집니다. (크기 16의 축이 없는) `4x4` slice만 가지고 있으므로, wraparound 연결이 없습니다. 따라서 대상 칩이 데이터를 받을 수 있는 포트가 2개 있고, 마찬가지로 소스 칩이 데이터를 보낼 수 있는 포트가 2개 있습니다. 전송해야 하는 데이터량은 `2 * 8 * 128 * 8192 = 1.7e7` 바이트입니다. 두 포트에서 동시에 전송할 수 있으므로 (즉, 배열의 절반은 오른쪽으로, 절반은 아래로 보내기), 초당 `2 * 4.5e10 = 9e10` 바이트를 전송할 수 있으며, 이는 전체 배열을 전송하는 데 약 `1.7e7 / 9e10 = 188us`가 걸린다는 것을 의미합니다 (대역폭 제한이라고 가정). `4x4` slice에서, 16개 미만의 칩을 가진 축에 대해서는 wraparound 링크가 없으므로 칩 $(0, 0)$과 $(3, 3)$ 사이에 6개의 hop이 있습니다. 각 hop의 지연 시간이 약 $1\mu s$이므로, 첫 번째 바이트는 약 `6us`에 도착하고 전체 전송은 `188us`가 걸립니다.

{% enddetails %}

**문제 6 [모든 것을 종합하기, 어려움]:** 큰 행렬 **A**: `int8[128 * 1024, 128 * 1024]`가 TPU v5e 4x4 slice에 균등하게 샤딩되어 있지만 각 칩의 호스트 DRAM에 오프로드되어 있다고 상상해보세요. 전체 배열을 TPU{0, 0}으로 복사하고 벡터 `bf16[8, 128 * 1024]`와 곱하고 싶다고 가정합시다. 이는 얼마나 걸릴까요? *힌트: 위의 숫자들을 사용하세요.*

{% details 답을 보려면 여기를 클릭하세요. %}

**답:** 수행해야 하는 연산들을 먼저 개요해봅시다. 우리 배열은 약 16GB입니다. 위의 표에서, TPU v5e 호스트는 4x2 토폴로지를 가지므로, 4x4는 2개의 호스트를 가집니다. 따라서, 배열이 균등하게 샤딩되어 있으므로, 각 호스트는 효과적으로 배열의 1/2인 8GB의 청크를 포함합니다. 이 청크들을 모두 TPU{0,0}으로 복사해야 하는데, 두 가지 옵션이 있습니다:

1. DCN을 통해 복사한 다음 전체 샤딩되지 않은 배열을 PCIe를 통해 HBM에 로드할 수 있습니다.
2. 샤딩된 배열을 해당 TPU에 로드한 다음, ICI를 통해 gather를 수행하고, TPU{0,0}에서 matmul을 수행할 수 있습니다.

옵션 (2)이 더 좋다는 것은 명확해야 합니다. DCN은 ICI에 비해 느리고 (호스트 0의 8개가 아닌) 많은 PCIe 링크를 통해 큰 배열을 로드하는 것을 훨씬 선호합니다. 다음은 시스템의 일부에 대한 다이어그램입니다. 위에서 설명한 바와 같이, TPU가 ICI에 의해 이웃에 연결되어 있고 (호스트를 가로질러서도), 모든 TPU가 (PCIe를 통해) 호스트 CPU에 연결되어 있으며, 호스트가 DCN에 의해 연결되어 있다는 점에 주목하세요.

{% include figure.liquid path="assets/img/challenge-problem.png" class="img-fluid img-small" caption="각 칩은 실제로 호스트에 대한 자체 PCIe 링크를 가지고 있지만, 명확성을 위해 하나만 여기에 표시되었습니다." %}

이제 각 부분이 얼마나 걸릴지 계산해봅시다:

1. **PCIe 로드**: 16개의 PCIe 링크를 통해 16GB / 2 = 8GB의 청크를 로드하고 있으며, 각각 `1.5e10` bytes/second 대역폭을 가집니다. 따라서 이는 약 33ms가 걸립니다.

2. **ICI 복사**: 각 TPU는 이제 우리 배열의 16GB / 16 = 1GB를 가집니다. ICI 대역폭은 링크당 *양방향* `9e10` bytes/second이고, 위의 다이어그램에서 TPU{0,0}에 대해 TPU v5e의 4개 ICI 링크 중 2개만 이 토폴로지에서 사용된다는 것을 알 수 있습니다. TPU{0,0}이 `4.5e10` bytes/s/링크로 2개 축을 따라 총 15GB를 받아야 하므로, `15e9 / (4.5e10 * 2) = 167ms`로 시간을 하한할 수 있습니다. 실제로는 로드가 매우 불균등하기 때문에 아마 달성 가능하지 않겠지만, 아마 2배 이내일 것입니다. 섹션 2에서 보게 될 것처럼, 전체 AllGather를 수행하는 것도 대략 `16e9 / (4.5e10 * 2)`가 걸리므로, 이는 최적에 가깝습니다.

3. **HBM $\rightarrow$ MXU 로드**: 최종 matmul을 수행하기 위해, 이 16e9 바이트와 bf16[8, 128 \* 1024] 배열(또 다른 2MB이므로 무시할 수 있음)을 HBM 대역폭을 통해 MXU에 로드해야 하며, 이는 `16e9 / 8.1e11 = 19ms`가 걸립니다.

4. **FLOP**: 총 $2 \cdot 8 \cdot 128 \cdot 1024 \cdot 128 \cdot 1024 = 2.7e11$ FLOP을 수행하고 있으며, `1.97e14` bf16 FLOP/s를 수행할 수 있으므로, 1.3ms를 얻습니다.

총 시간의 상한은 이 모든 시간의 합이지만, TPU는 일반적으로 이러한 연산들을 겹칠 수 있으므로, 이를 가장 느린 부분에 의해 병목되는 파이프라이닝 문제로 생각할 수 있습니다. 그것이 사실이라고 가정하면, 답은 약 150-200ms입니다.

{% enddetails %}

<h3 markdown=1 class="next-section">Part 2는 여기까지입니다! 분할과 cross-TPU 통신을 다루는 Part 3은 [여기를 클릭하세요](../sharding).</h3>

## 부록

### 부록 A: GPU에 대한 모든 것

Volta 세대(V100) 이후로, TPU와 GPU는 매우 비슷해 보이기 시작했습니다: _둘 다 행렬 곱셈을 매우 빠르게 하는 것을 목표로 합니다_. 둘 다 CPU에 연결된 accelerator 역할을 하며 많은 구성 요소가 대략 유사합니다 (모든 용어를 몰라도 걱정하지 마세요, 나중에 모두 소개할 것입니다):

|     TPU     |                    GPU                    |
| :---------: | :---------------------------------------: |
| Tensor Core |      SM ("Streaming Multiprocessor")      |
|     HBM     |                   DRAM                    |
|    VMEM     |     SMEM (often used as an L1 cache)      |
|     VPU     | Warp scheduler (a set of SIMD CUDA cores) |
|     MXU     |                Tensor Core                |
|     ICI     |              NVLink/NVSwitch              |

GPU의 핵심 유닛은 SM 또는 "streaming multiprocessor"로, 위에서 설명한 전체 TPU Tensor Core와 대략 유사합니다. 하지만 TPU에 비해 GPU는 _훨씬 더 많은_ 것들을 가집니다 (H100은 약 144개). 각 SM은 TPU MXU처럼 작동하는 자체 행렬 곱셈 유닛(혼란스럽게도 Tensor Core라고 함)과 TPU VPU처럼 작동하는 Warp scheduler라고 하는 4개의 좁은 SIMD 유닛 세트(32 레인 대신 1024)를 가집니다. 더 독립적인 SM은 계산을 더 유연하게 만들지만(각각이 완전히 독립적인 작업을 할 수 있으므로) 하드웨어를 더 비싸고 추론하기 복잡하게 만듭니다.

{% include figure.liquid path="assets/img/b100-sm-diagram.png" class="img-small" caption="<b>그림:</b> Blackwell (B100) SM의 기본 구성 요소. 다이어그램은 4개의 SIMD compute unit(warp scheduler라고 함)을 보여주며, 각각 행렬 곱셈을 위한 Tensor Core를 가집니다. 이는 또한 warp scheduler당 레지스터, SM 수준 L1 캐시, 그리고 Blackwell의 새로운 추가인 TMEM 또는 tensor 메모리를 보여줍니다." %}

각 SM은 또한 데이터 접근을 가속화하고 레지스터 spilling을 위해 사용되는 O(256kB) L1 캐시(SMEM이라고도 함)를 가집니다. L1 캐시에 사용되는 메모리의 섹션은 또한 thread-block의 모든 스레드에서 접근을 허용하는 shared memory로 선언될 수 있으며, 사용자 정의 캐시, 병렬 reduction 및 동기화 등에 사용됩니다 (TPU의 VMEM과 유사).

GPU는 또한 모든 SM에서 공유되는 추가 L2 캐시를 가집니다. VMEM과 달리, 이는 하드웨어가 관리하며 캐시 히트를 최적화하는 것이 성능에 종종 중요합니다.

**네트워킹:**

* 주요 차이점은 NVIDIA GPU가 일반적으로 스위치(NVLink $\rightarrow$ NVSwitch)를 통해 8-256개 GPU의 'clique'에 있다는 것으로, 이는 해당 'clique' 내의 모든 GPU 간에 point-to-point 통신을 허용하지만, 256개보다 많은 GPU 간의 통신은 현저히 느리다는 것을 의미합니다 - 이는 256개보다 많은 훈련이 일반적으로 더 복잡한 pipeline 병렬성을 요구한다는 것을 의미합니다 (대조적으로, PaLM은 각각 3072개의 TPU 칩을 가진 두 개의 clique에서 훈련되었습니다).
* AllReduce와 같은 일반적인 신경망 연산의 경우, all-to-all 연결은 이점을 가지지 않지만 (같은 통신 패턴이 상관없이 발생해야 하므로), MoE 모델을 더 많은 GPU에 걸쳐 저장하고 expert를 더 효율적으로 전송할 수 있게 해줍니다.
* 각 GPU는 GPU 자체와 비슷한 비용의 스위치를 필요로 하므로, ICI와 같은 on chip interconnect가 더 저렴합니다.
* [NVIDIA deep learning performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch)
* [NVSwitch](https://www.nvidia.com/en-au/data-center/nvlink/)
* 매우 다른 Tensor Parallelism / Pipeline Parallelism 전환점!

### 부록 B: Systolic array는 어떻게 작동하는가?

TPU MXU의 핵심에는 `128x128` systolic array가 있습니다 (TPU v6e에서는 `256x256`). 완전히 포화될 때 systolic array는 8 클록 사이클마다 하나의 `bfloat16[8,128] @ bf16[128x128] -> f32[8,128]`<d-footnote>이 표기법에 익숙하지 않다면, bfloat16 요소를 가진 `8x128` 행렬에 bfloat16 요소를 가진 `128x128` 행렬을 곱하고 결과를 float32 요소를 가진 `8x128` 행렬에 저장한다는 의미입니다.</d-footnote> 곱셈을 수행할 수 있습니다.

* 핵심적으로, systolic array는 각각 곱셈과 덧셈 연산을 수행할 수 있는 ALU의 2D `128x128` (`=16,384`) 격자입니다.
* 가중치(**W**, `128x128` 입력)는 위에서 아래로 전달되고 (RHS라고 함) 입력(**X**, `8x128` 입력)은 왼쪽에서 전달됩니다 (LHS라고 함).

다음은 가중치 세트(파란색)와 activation 세트(녹색)를 곱하는 단순화된 애니메이션입니다. 가중치(RHS)가 먼저 대각선으로 부분적으로 로드된 다음, activation이 역시 대각선으로 공급되는 것을 알 수 있습니다. 아래의 각 프레임에서, 겹쳐진 모든 녹색과 파란색 유닛을 곱하고, 위에서 전달된 잔여와 결과를 합한 다음, 결과를 한 유닛 아래로 차례로 전달합니다.

{% include figure.liquid path="assets/img/systolic-array.gif" %}

다음은 계산에서 출력이 스트리밍되는 것을 보여주는 이 애니메이션의 더 일반적인 버전입니다:

{% include figure.liquid path="assets/img/systolic-array2.gif" class="img-small" %}

다음은 여러 RHS와 LHS 배열에 걸쳐 파이프라인될 수 있는 방법을 보여주는 다이어그램입니다:

{% include figure.liquid path="assets/img/systolic-array-pipelining.png" class="img-fluid" %}

가중치(RHS)와 activation(LHS)이 로드될 때 초기 파이프라인 버블이 있습니다. 그 초기 버블 후에는, 새로운 입력과 가중치가 추가 버블 없이 로드될 수 있습니다.

다음은 bf16[2, 3] x bf16[3, 3] 행렬 곱셈의 나쁜 애니메이션으로, 이를 batch 1과 크기 3의 입력 activation을 가진 2x3 가중치 행렬의 matmul로 상상할 수 있습니다. 이는 이전 슬라이드에 비해 회전되어 있고 입력이 아래가 아닌 오른쪽으로 흘러나가지만, 대략적인 구조를 볼 수 있습니다.

{% include figure.liquid path="assets/img/systolic-array-bad.gif" class="img-small" %}

너무 큰 파이프라인 버블 없이 큰 행렬을 곱하기 위해 이를 효율적으로 파이프라인할 수 있습니다. 그렇긴 하지만, 행렬이 일반적으로 128x128인 MXU의 측면 차원보다 큰 모양을 가지는 것이 중요합니다. 일부 TPU(TPU v3 이후)는 여러 MXU를 가지므로 (TPU v3의 경우 2개, TPU v4/5의 경우 4개), 타일링 차원이 128 * MXU 수보다 큰지 확인해야 합니다. [여기](https://www.youtube.com/watch?v=sJltBQ4MOHA)에 이에 대한 좋은 애니메이션이 있습니다.

Trillium (TPU v6e)은 `256x256` systolic array를 가지므로, 사이클당 4배 더 많은 FLOP을 수행할 수 있습니다. 이는 또한 MXU를 완전히 활용하기 위해 tensor의 차원이 두 배 커야 함을 의미합니다.

[이 블로그 포스트](https://fleetwood.dev/posts/domain-specific-architectures#google-tpu)에는 고정 가중치 행렬에 대한 systolic array 곱셈의 또 다른 훌륭한 애니메이션이 있습니다.

### 부록 C: TPU 내부에 대한 자세한 내용

### Scalar Core

Scalar core는 TPU의 제어 유닛입니다. 모든 명령을 fetch하고 dispatch하며 HBM에서 VMEM으로의 전송을 실행하고, scalar metadata 작업을 하도록 프로그래밍될 수 있습니다. Scalar core가 single-threaded이기 때문에, 이것의 부작용은 TPU의 각 core가 사이클당 하나의 DMA 요청만 생성할 수 있다는 것입니다.

이를 맥락에서 보면, 단일 scalar core가 2048개의 ALU, 4개의 MXU, 2개의 XLU, 그리고 여러 DMA 엔진으로 구성된 VPU를 제어합니다. compute unit당 제어의 고도로 치우친 특성은 하드웨어 효율성의 원천이지만, 흥미로운 방식으로 데이터 의존적 벡터화를 수행하는 능력을 제한합니다.

### VPU

TPU vector core는 vadd(벡터 덧셈)나 vmax(elementwise max)와 같은 벡터 연산을 수행하는 2차원 SIMD 벡터 머신(**VPU**)과 VPU와 MXU를 위한 데이터를 보유하는 **VREG**라고 하는 벡터 레지스터 세트로 구성됩니다. v5p의 각 TPU core는 64개의 32비트 VREG(v4에서는 32개)를 가져서, 총 약 `64 * 8 * 128 * 4 = 256kB`의 VREG 메모리를 제공합니다.

VPU는 효과적으로 `(8, 128)` 모양의 2D 벡터 산술 유닛으로, 128 차원은 lane axis라고 하고 8 차원은 sublane axis라고 합니다. v5의 각 (lane, sublane) 쌍은 4개의 표준 부동 소수점 및 정수 ALU를 포함합니다. 소프트웨어 관점에서, 이는 v5에서 총 4048개의 부동 소수점 가산기를 가진 8x128 벡터 유닛의 모습을 만듭니다.

VPU는 대부분의 산술 명령을 각 ALU에서 2 사이클의 지연 시간으로 한 사이클에 실행하므로, 예를 들어 v5에서는 매 사이클마다 VREG에서 4쌍의 f32 값을 더할 수 있습니다. 일반적인 VPU 명령은 `{v2 = vadd.8x128.f32 v0, v1}`처럼 보일 수 있으며, 여기서 v0와 v1은 입력 VREG이고 v2는 출력 VREG입니다.

모든 lane과 sublane은 순수한 SIMD 방식으로 매 사이클마다 같은 프로그램을 실행하지만, 각 ALU는 다른 연산을 수행할 수 있습니다. 따라서 예를 들어 단일 사이클에서 1개의 vadd와 1개의 vsub를 처리할 수 있으며, 각각은 두 개의 전체 VREG에서 작동하고 출력을 세 번째에 씁니다.

Lane 내의 reduction(크기 8 sublane 차원에 걸쳐)은 저렴하고 매우 효율적입니다 (3번의 순열과 3번의 덧셈). Cross-lane reduction은 더 어렵고 "cross lane unit"인 XLU를 포함하며, 이는 느리고 상당히 비쌉니다.