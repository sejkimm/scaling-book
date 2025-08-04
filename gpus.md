---
layout: distill
title: "GPU에 대해 생각하는 방법"
description: "TPU에 대해 그렇게 많이 이야기한 후, 나머지 세계의 많은 부분이 사용하는 것인 NVIDIA GPU를 살펴볼 가치가 있을 것입니다. 이는 현대 NVIDIA ML GPU(예: H100 또는 B100)의 chip과 networking 수준 모두에 대한 심층 탐구이며, 어떤 종류의 LLM parallelism을 허용하는지 알아보겠습니다. 이 책의 나머지 부분을 읽은 후에 이를 읽는 것을 권합니다."
date: 2025-07-25
future: true
htmlwidgets: true
hidden: false

section_number: 12

previous_section_url: 
previous_section_name: ...

next_section_url:
next_section_name: ...

bibliography: main.bib

giscus_comments: true

authors:
  - name: To Be Determined
    url: "https://www.jacobaustin.org/"
    affiliations:
      name: Google DeepMind

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: GPU란 무엇인가?
  - subsections:
    - name: "GPU 사양 요약"
    - name: "Grace Hopper"
    - name: Chip 수준에서 GPU vs. TPU
    - name: 실습 문제
  - name: Networking
  - subsections:
    - name: Node 수준
    - name: 실습 문제
    - name: Node 수준을 넘어서
  - name: GPU에서 Collective는 어떻게 작동하는가?
  - subsections:
    - name: Node 내에서
    - name: In network reduction
    - name: Node 수준을 넘어서
    - name: 실습 문제
  - name: "GPU에서 LLM Scaling을 위한 핵심 사항"
  - subsections:
    - name: 실습 문제
  - name: "B100과 GB200 NVL72에서는 어떻게 달라지는가?"

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

GPU는 비디오 게임을 렌더링하기 위한 전용 하드웨어로 시작되었지만, 2010년대 AI 수요의 폭발 이후 점점 더 전용 matrix multiplication 머신처럼 보이기 시작했습니다 – 다시 말해, TPU와 똑같습니다. TPU와 GPU 모두 CPU에 연결된 matrix multiplication accelerator처럼 작동합니다. 이들은 두 가지 핵심 측면에서 다릅니다: 어떻게 네트워킹되어 있는지, 그리고 올바른 일을 하기 위해 소프트웨어에 얼마나 많은 책임을 부여하는지 (*힌트: TPU는 컴파일러가 훨씬 더 많은 작업을 해야 합니다*).

## GPU란 무엇인가?

현대 GPU(예: H100, B100)는 기본적으로 matrix multiplication에 특화된 compute core들(Streaming Multiprocessor 또는 SM이라고 함)이 fast memory stick(DRAM 또는 HBM이라고 함)에 모두 연결된 것입니다. 다음은 diagram입니다:

{% include figure.liquid path="assets/gpu/gpu-diagram.png" class="img-fluid" caption="<b>그림:</b> 현대 NVIDIA GPU의 기본 구성 요소. diagram은 Tensor Core와 Warp Scheduler(CUDA core 포함), SMEM, 공유 L2 cache, 그리고 main GPU memory(HBM)를 포함하는 SM을 보여줍니다." %}

최대 2개의 Tensor Core를 가진 TPU와 달리, **현대 GPU는 100개 이상의 이러한 SM을 가집니다**(H100에서는 132개). 따라서 이러한 각 SM은 TPU TensorCore보다 훨씬 덜 강력하지만 전체 시스템은 더 유연합니다. 각 SM은 거의 완전히 독립적이므로 GPU는 수백 개의 작업을 동시에 할 수 있지만, 모든 SM은 50MB L2 cache와 대용량 DRAM(H100에서 80GB, B100에서 192GB)을 공유합니다. 위에 표시된 것처럼입니다. 다음은 B100 SM의 더 자세한 view입니다:

{% include figure.liquid path="assets/gpu/broadwell-sm.png" class="img-small" caption="<b>그림:</b> 단일 Broadwell(B100) SM에 대한 더 깊은 관찰로, 4개의 SM subpartition과 그들의 CUDA core, 4개의 TensorCore 및 일부 보조 유닛을 보여줍니다." %}

각 SM은 4개의 동일한 quadrant로 나뉘는데, NVIDIA가 "SM subpartition"이라고 부르며, 각각 Tensor Core, 16k 32-bit register, 그리고 NVIDIA가 "CUDA Core"라고 부르는 SIMD lane 집합을 포함합니다. 각 partition의 핵심 구성 요소는 아마도 matrix multiplication을 수행하고 FLOP/s의 대부분을 차지하는 Tensor Core이지만, TPU처럼 주목할 가치가 있는 몇 가지 다른 구성 요소가 있습니다.

* **CUDA Core:** 각 subpartition은 TPU의 VPU와 매우 유사하게 SIMD vector arithmetic을 수행하는 **CUDA Core**라고 불리는 ALU 집합을 포함합니다. 각 subpartition은 32개의 fp32 core(그리고 더 적은 수의 int32와 fp64 core)를 포함하며, 이들은 각 cycle에서 동일한 vector operation을 수행하는 단일 SIMD unit로 기능합니다. **NVIDIA의 의심스러운 명명에도 불구하고, 이들을 각 cycle에서 동일한 vector operation을 수행하는 32-wide SIMD unit의 lane으로 생각해야 합니다.**

    * subpartition 내의 특정 precision의 각 CUDA core(SIMD lane)는 lockstep으로 실행되므로, cycle당 각 lane은 TPU의 VPU와 마찬가지로 동일한 작업을 수행해야 합니다.
    * ML model의 경우, CUDA core는 일반적으로 ReLU, vector addition, norm, 그리고 기타 non-matmul 작업과 같은 pointwise operation을 수행하는 데 사용됩니다.
    * subpartition 내의 CUDA core는 SIMD unit의 controller처럼 작동하는 **warp scheduler**라고 불리는 dispatch unit에 의해 제어됩니다. 각 warp scheduler는 많은 프로그램(**warp**라고 함)을 동시에 실행할 수 있지만(subpartition당 최대 16개) 각 clock cycle에서는 단일 프로그램의 instruction만 실행한다는 점에서 multi-threaded CPU와 비슷하게 실행됩니다. 하드웨어는 memory load와 같은 I/O operation을 숨기기 위해 active warp 간에 자동으로 전환합니다.
    * 각 warp scheduler는 자체 register file을 가집니다(H100/B100에서 16,384개의 32-bit word, SM당 총 `4 * 16384 * 4 = 256kB`의 register memory). 각 CUDA core는 최대 256개의 register에만 접근할 수 있으므로, warp scheduler당 최대 16개의 "resident warp"를 스케줄링할 수 있지만, 각 core가 256개의 register를 사용한다면 한 번에 2개만 맞출 수 있습니다.

* **Tensor Core (TC):** 각 subpartition은 TPU MXU와 같은 전용 matrix multiplication unit인 자체 Tensor Core를 가집니다. Tensor Core는 GPU FLOP/s의 대부분을 나타냅니다(예: H100에서 CUDA core의 66 FLOP/s에 비해 990 bf16 FLOP/s).

    * H100은 초당 990 bfloat16 TFLOP의 peak bfloat16 matmul throughput을 가지므로, 각 SM은 약 7.5TFLOP peak를 할 수 있습니다. 각 SM이 4개의 TC를 가지고 각 SM이 1.76GHz의 peak frequency에서 실행되므로, 각 TC는 대략 `7.5e12 / 1.76e9 / 4 ~ 1024` bf16 FLOP/s를 할 수 있으므로, 각 cycle당 대략 `8x8x8` matmul입니다.
    * 각 GPU 세대는 matrix multiplication compute가 점점 더 중요해지면서 더 큰 Tensor Core를 가지게 되었습니다([이에 대한 좋은 기사](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)).
    * TPU처럼, GPU는 더 높은 throughput으로 더 낮은 precision matmul을 할 수 있습니다. H100은 990 bf16 TFLOP/s와 1979 fp8 TFLOP/s를 할 수 있어서, 약 2배입니다. 이는 더 낮은 precision에서 model을 효과적으로 training하거나 serving할 수 있다면 성능에서 상당한 향상을 볼 것임을 의미합니다.
    * 역사적으로, GPU Tensor Core는 SMEM이나 register memory에서 입력을 로드했지만, TC가 커지면서 전체 입력을 맞추기가 어려워졌습니다. B100/B200은 TC에서 수행되는 matmul의 입력을 저장하는 데 사용되는 Tensor Memory(또는 TMEM)라고 불리는 새로운 종류의 on-chip memory를 도입합니다.

compute unit 외에도, GPU는 가장 큰 것이 HBM(main GPU memory)이고, 그 다음 일련의 더 작은 cache(L2, L1, TMEM, register memory)인 memory 계층을 가집니다.

* **SMEM (L1 Cache):** 각 SM은 SMEM이라고 불리는 자체 작은 on-chip cache를 가지는데, 이는 "shared memory"로 프로그래머가 제어하거나 하드웨어가 on-chip cache로 사용할 수 있습니다.

    * SMEM은 TC matmul의 입력을 저장하고 처리되는 동안 activation을 저장하는 데 많이 사용되므로, DRAM에서 완전히 로드할 필요가 없습니다.
    * SMEM이 TPU VMEM보다 훨씬 작기 때문에, model의 전체 layer를 on-chip memory에 맞추기가 더 어렵습니다.

* **L2 Cache:** 모든 SM은 main memory 접근을 줄이는 데 사용되는 비교적 큰 ~50MB L2 cache를 공유합니다.

    * 이는 TPU의 VMEM과 크기가 비슷하며, HBM보다 TC에 대한 bandwidth가 높지만, 프로그래머가 제어하지 않고 **훨씬** 느립니다. 이는 프로그래머가 L2 cache가 잘 사용되도록 memory 접근 패턴을 수정해야 하는 약간의 "spooky action at a distance"를 만듭니다.
    * NVIDIA는 자사 chip의 L2 bandwidth를 공개하지 않지만, 약 5.5TB/s로 측정되었으며, 이는 HBM bandwidth의 대략 1.6배입니다. 비교하자면, TPU의 VMEM은 2배 더 크고 훨씬 더 많은 bandwidth(약 40TB/s)를 가집니다.

* **HBM:** model weight, gradient, activation 등을 저장하는 데 사용되는 main GPU memory.

    * HBM 크기는 Volta의 32GB에서 Blackwell(B100)의 192GB로 많이 증가했습니다.
    * HBM에서 CUDA Tensor Core로의 bandwidth는 HBM bandwidth 또는 memory bandwidth라고 불리며, H100에서 약 3.35TB/s입니다.

다음은 GPU와 TPU 구성 요소를 비교하는 유용한 cheat sheet입니다:

|              GPU              |     TPU     |              무엇인가?              |
| :---------------------------: | :---------: | :-----------------------------------: |
| Streaming Multiprocessor (SM) | Tensor Core | 다른 unit을 포함하는 핵심 "cell" |
|        Warp Scheduler         |     VPU     |      SIMD vector arithmetic unit      |
|           CUDA core           |  VPU lane   |            SIMD ALU "lane"            |
|        SMEM (L1 Cache)        |    VMEM     |       Fast on-chip cache memory       |
|          Tensor Core          |     MXU     |      Matrix multiplication unit       |
|             DRAM              |     HBM     |  High bandwidth high capacity memory  |

### GPU 사양 요약

다음은 최근 model의 GPU 사양 요약입니다:

|  GPU  | Generation |  SMs  | SMEM per SM (kB) | L2 Cache (MB) | Clock Speed (GHz) | DRAM (GB) | DRAM BW (TB/s) | BF16 TFLOPs | FP8 TFLOPs | FP4 TFLOPs |
| :---: | :--------: | :---: | :--------------: | :-----------: | :---------------: | :-------: | :------------: | :---------: | :--------: | :--------: |
| V100  |   Volta    |  80   |        96        |       6       |     1.25/1.38     |    32     |      0.9       |      —      |     —      |     —      |
| A100  |   Ampere   |  108  |       192        |      40       |     1.10/1.41     |    80     |      2.0       |     312     |     —      |     —      |
| H100  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    80     |      3.35      |     990     |    1979    |     —      |
| H200  |   Hopper   |  132  |       256        |      50       |     1.59/1.76     |    141    |      4.8       |   (same)    |   (same)   |     —      |
| B100  | Blackwell  |  144  |       256        |      50       |     1.67/1.83     |    192    |       8        |    1800     |    3500    |    7000    |
| B200  | Blackwell  |   ?   |       256        |      50       |         ?         |    192    |       8        |    2250     |    4500    |    9000    |

모든 세대는 SM당 256kB의 register memory를 가집니다. Blackwell은 또한 SM당 256kB의 TMEM을 추가합니다. 일부 사양은 GPU의 정확한 버전에 따라 약간 다릅니다.

### Grace Hopper

NVIDIA는 또한 일부 GPU를 Grace Hopper CPU와 쌍을 이루는 GH200과 GB200 시스템을 판매합니다. 예를 들어, GH200은 1개의 H200과 1개의 GH CPU를 가지고, GB200 시스템은 2개의 B200과 1개의 GH CPU를 가집니다. 이 시스템의 장점은 CPU가 full bandwidth NVLink 연결(NVLink C2C라고 함)을 사용하여 GPU에 연결되므로, parameter를 offload하는 데 유용한 매우 높은 CPU to GPU bandwidth를 가진다는 것입니다. 다시 말해, 주어진 GPU에 대해 host memory에 도달하는 bandwidth는 다른 GPU의 HBM에 도달하는 것과 동일합니다.

### Chip 수준에서 GPU vs. TPU

보시다시피, GPU와 TPU는 chip 수준에서 매우 유사해 보입니다. 둘 다 matmul accelerator, SIMD vector unit, cache memory를 가집니다. 한 가지 주요 차이점은 TPU가 1-2개의 큰 Tensor Core를 가지는 반면, GPU는 수백 개의 작은 SM을 가진다는 것입니다. 마찬가지로, 각 Tensor Core는 4096개의 ALU를 가진 1개의 큰 VPU를 가지는 반면, GPU H100은 132 * 4 = 528개의 작고 독립적인 SIMD unit을 가집니다.

다음은 GPU와 TPU의 1:1 비교입니다:

| GPU                           | TPU                      | 몇 개인가?                                                                                           |
| :---------------------------- | :----------------------- | :-------------------------------------------------------------------------------------------------- |
| SM (streaming multiprocessor) | Tensor Core              | H100은 132개, TPU는 1-2개.                                                                          |
| Warp scheduler                | VPU                      | 각 SM은 4개, 따라서 H100에서 132 * 4 = 528개. TPU v5는 효과적으로 Tensor Core당 4개, 총 8개.      |
| SMEM (L1 cache)               | VMEM                     | GPU는 SM당 256kB, 총 ~32MB. TPU는 훨씬 더 높은 bandwidth로 총 약 120MB의 VMEM. |
| Register                     | Vector Register (VReg) | GPU는 SM당 256kB, TPU는 총 256kB.                                                            |
| Tensor Core                   | MXU                      | TPU v5p는 TC당 4개의 MXU, 총 8개. 각 H100 SM은 4개, 총 528개.                        |

보시다시피, TPU는 GPU보다 훨씬 덜 modular합니다! 이는 구축하기에는 더 저렴하지만 프로그래밍하기에는 더 복잡하게 만듭니다. 예를 들어, TPU는 matmul이 core `[8, 128] x [128, 128]` 크기의 배수여야 하고 vector 작업이 `[8, 128]`의 증분으로 수행되어야 하며, 입력 배열을 이 크기로 pad합니다. TPU는 또한 예를 들어 vector arithmetic이 주어진 matrix multiplication보다 오래 걸리는 경우 stall하기 쉬운데, 각각 1개씩만 있고 종종 함께 fuse되기 때문입니다. 하지만 제대로 사용하면, GPU의 modular 특성에서 오는 엄청난 비용과 하드웨어 복잡성을 피할 수도 있습니다.

TPU는 또한 weight와 activation을 저장하는 데 사용할 수 있는 훨씬 더 많은 fast cache memory를 가집니다. 이는 weight를 VMEM으로 지속적으로 fetch할 수 있다면 LLM inference에서 더 빠르게 만들 수 있습니다.

### 실습 문제

다음은 위 내용의 일부를 테스트하는 문제들입니다. 답이 제공되지만, 보기 전에 펜과 종이를 손에 들고 질문에 답하려고 시도하는 것이 좋은 아이디어일 것입니다.

**문제 1 [CUDA core]:** H100은 몇 개의 CUDA core를 가지고 있나요? 이것이 TPU v5p의 독립적인 lane 수와 어떻게 비교되나요?

{% details 답을 보려면 클릭하세요. %}

**답:** `132 * 32 * 4 = 16896` CUDA core. TPU v5p는 2개의 TensorCore(보통 Megacore를 통해 연결됨)를 가지며, 각각 (8, 128) lane과 lane당 4개의 독립적인 ALU를 가진 VPU를 가지므로, 2 * 4 * 8 * 128 = 8192입니다. 이는 대략 같은 frequency에서 실행되는 H100의 vector lane 수의 절반입니다.

{% enddetails %}

**문제 2 [Vector FLOP 계산]:** 단일 H100은 132개의 SM을 가지고 1.59GHz의 clock speed(최대 1.98GHz boost)에서 실행됩니다. thread당 cycle당 하나의 vector op을 할 수 있다고 가정하세요. 초당 몇 개의 vector FP32 FLOP을 할 수 있나요? boost로는? 이것이 matmul FLOP과 어떻게 비교되나요?

{% details 답을 보려면 클릭하세요. %}

**답:** `132 * 32 * 4 * 1.59e9 = 26.9` TFLOP/s. boost로는 33.5 TFLOP/s. 이는 기술적으로 한 cycle에서 두 FLOP으로 계산되는 FMA(fused-multiply-add)를 할 수 있지만 이는 기본적으로 달성할 수 없기 때문에 [spec sheet](https://www.nvidia.com/en-us/data-center/h100/)에서 보고된 것의 절반입니다. 990 bfloat16 matmul TFLOP/s를 할 수 있으므로, FMA를 무시하면 Tensor Core는 약 30배 더 많은 FLOP/s를 합니다.

{% enddetails %}

**문제 3 [GPU matmul intensity]:** H100에서 peak bf16 matmul intensity는 무엇인가요? B200은?

{% details 답을 보려면 클릭하세요. %}

**답:** H100의 경우, peak 990e12 bf16 FLOP과 3.35e12 byte/s의 bandwidth를 가집니다. 따라서 critical intensity는 990e12 / 3.35e12 = 295로, TPU의 240과 상당히 유사합니다. B200의 경우 2250e12 / 8e12 = 281로, 매우 유사합니다. 이는 TPU와 마찬가지로 matmul에서 compute-bound가 되려면 약 280의 batch size가 필요함을 의미합니다.

{% enddetails %}

**문제 4 [L1 cache capacity]:** H100의 총 L1 cache/SMEM capacity는 무엇인가요? register memory는? 이것이 TPU VMEM capacity와 어떻게 비교되나요?

{% details 답을 보려면 클릭하세요. %}

**답:** SM당 256kB를 가지므로, 각각 약 33MB, 총 약 66MB입니다. 이는 현대 TPU VMEM의 120MB의 약 절반이지만, TPU는 총 256kB의 register memory만 가집니다! TPU VMEM latency가 SMEM latency보다 낮은데, 이는 GPU에서 그렇게 적은 register memory로도 해낼 수 있는 이유 중 하나입니다.

{% enddetails %}

## Networking

Networking은 아마도 GPU와 TPU가 가장 다른 영역일 것입니다. 보았듯이, TPU는 각 TPU가 이웃에만 연결된 2D 또는 3D tori로 연결됩니다. 이는 두 TPU 간에 메시지를 보내는 것이 모든 중간 TPU를 통과해야 함을 의미하고, mesh에 대해 uniform communication pattern만 사용하도록 강제합니다. 일부 측면에서는 불편하지만, 이는 또한 TPU당 link 수가 일정하고 임의로 큰 TPU "pod"로 확장할 수 있음을 의미합니다.

반면 GPU는 더 전통적인 hierarchical tree-based switching network를 사용합니다. **node**라고 불리는 8개 GPU의 집합(B200의 경우 최대 72개)이 매우 높은 bandwidth로 서로 1 hop 내에 연결되고, 이러한 node는 network switch(NVSwitch라고 브랜드됨)로 더 큰 unit(SU 또는 scalable unit이라고 함)에 연결되며, 이는 차례로 higher level switch로 더 큰 unit에 연결됩니다.

### Node 수준

GPU node는 일반적으로 8개 GPU(B200의 경우 최대 72개)의 작은 unit으로, all-to-all, full-bandwidth connectivity를 가집니다. 각 node는 고대역폭 Infiniband NVLink로 모든 local GPU에 연결된 일정 수의 NVSwitch를 가집니다.

실제 node-level topology는 시간이 지나면서 node당 switch 수를 포함하여 상당히 변했지만, H100의 경우 node당 4개의 NVSwitch를 가지며, GPU가 5 + 4 + 4 + 5 link 패턴으로 연결됩니다:

{% include figure.liquid path="assets/gpu/nvlink-nodes.png" class="img-fluid" caption="<b>그림:</b> 다양한 NVIDIA GPU 세대가 GPU를 node로 연결하는 방법. networking 구성과 node당 NVSwitch 수가 세대마다 변했음에 주목하세요." %}

H100의 경우, 각 NVLink link는 각 방향당 25GB/s의 bandwidth(B100의 경우 50GB/s)를 가지며, 각 GPU에서 network로 18 * 25=450GB/s의 full-duplex bandwidth를 제공합니다. 이러한 거대한 switch는 최대 64개의 NVLink port를 가지므로, 4개의 switch를 가진 H100의 경우 총 64 * 25e9 * 4=6.4TB/s의 bandwidth를 처리할 수 있습니다.

{% include figure.liquid path="assets/gpu/nvlink4.png" class="img-fluid" caption="<b>그림:</b> 단일 NVLink4 Switch가 어떻게 작동하는지 보여주는 NVIDIA 판매 diagram(각각 50GB/s의 bandwidth를 가진 64개의 port 포함)." %}

다음은 이러한 수치가 GPU 세대에 따라 어떻게 변했는지에 대한 개요입니다:

| NVLink Gen | NVSwitch Gen | GPU Generation | NVLink Bandwidth (GB/s, full-duplex) | Max Links / GPU | Node GPU to GPU bandwidth (GB/s full-duplex) | Node size (NVSwitch domain)         |   NVSwitches per node    |
| :--------: | :----------: | :-------------: | :----------------------------------: | :-------------: | :------------------------------------------: | :----------------------------------: | :----------------------: |
|  **3.0**   |   **2.0**    | Ampere         |                  25                  |       12        |                     300                      | 8                                   |            6             |
|  **4.0**   |   **3.0**    | Hopper         |                  25                  |       18        |                     450                      | 8                                   |            4             |
|  **5.0**   |   **4.0**    | Blackwell      |                  50                  |       18        |                     900                      | 8 for H200/B200, 72 for GB200 NVL72 | 2 for B200, 18 for NVL72 |

### 실습 문제

다음은 networking에 대한 몇 가지 더 많은 Q/A 문제입니다. 실제 communication pattern을 다루게 하므로 이를 해보는 것이 특히 유용하다고 생각합니다.

**문제 1 [H100 node의 총 bandwidth]:** 4개의 switch를 가진 8xH100 node에서 node당 얼마나 많은 총 bandwidth를 가지나요? *힌트:* NVLink와 NVSwitch bandwidth를 모두 고려하세요.

{% details 답을 보려면 클릭하세요. %}

**답:** Gen4 4xNVSwitch를 가지며, 각각 64*25e9=1.6TB/s의 단방향 bandwidth를 가집니다. 그러면 switch 수준에서 4 * 1.6e12=6.4e12 bandwidth를 가질 것입니다. 하지만 각 GPU는 450GB/s의 단방향 bandwidth만 처리할 수 있으므로, 최대 450e9 * 8 = 3.6TB/s bandwidth를 가짐을 의미합니다. 이것이 더 작으므로, peak bandwidth는 3.6TB/s입니다.

{% enddetails %}

**문제 2 [Bisection bandwidth]:** Bisection bandwidth는 network의 임의 균등 분할 간에 사용 가능한 가장 작은 bandwidth로 정의됩니다. 다시 말해, network를 두 개의 동일한 절반으로 나누면, 두 절반 사이에 얼마나 많은 bandwidth가 교차하나요? 8x H100 node의 bisection bandwidth를 계산할 수 있나요? *힌트:* bisection bandwidth는 일반적으로 양방향 flow를 포함합니다.

{% details 답을 보려면 클릭하세요. %}

**답:** 임의의 균등 분할은 각 절반에 4개의 GPU를 가지며, 각각은 다른 절반으로 4 * 450GB/s를 egress할 수 있습니다. 양방향 flow를 고려하면, 8 * 450GB/s의 byte가 분할을 교차하거나, 3.6TB/s의 bisection bandwidth를 제공합니다. 이는 NVIDIA가 예를 들어 [여기](https://hc34.hotchips.org/assets/program/conference/day2/Network%20and%20Switches/NVSwitch%20HotChips%202022%20r5.pdf)에서 보고하는 것입니다.

{% enddetails %}

**문제 3 [AllGather 비용]:** B byte의 배열이 주어졌을 때, 8xH100 node에서 (throughput-bound) AllGather가 얼마나 걸릴까요? D=1024, F=16,384인 bf16[DX, F]에 대한 계산을 하세요. *이를 답하기 전에 TPU collective [섹션](https://jax-ml.github.io/scaling-book/sharding/)을 읽어볼 가치가 있습니다. 여기서 이를 생각해보되 다음에 collective에 대해 훨씬 더 많이 이야기하겠습니다.*

{% details 답을 보려면 클릭하세요. %}

**답:** 각 GPU는 450GB/s를 egress할 수 있고, 각 GPU는 $B / N$ byte(여기서 N=8, node 크기)를 가집니다. 각 node가 다른 $N - 1$ node에 차례로 byte를 보내는 것을 상상할 수 있으며, 총 $N - 1$번의 turn으로 각각 $T_\text{comms} = (B / (N * W_\text{unidirectional}))$이거나, $T_\text{comms} = (N - 1) * B / (N * W_\text{unidirectional})$입니다. 이는 대략 $B / (N * W_\text{uni})$ 또는 $B$ / 3.6e12, bisection bandwidth입니다.

주어진 배열의 경우, B = `1024*16384*2=32MB`이므로, 총 시간은 `33.5e6 / 3.6e12 = 9us`입니다. 이는 latency-bound일 수 있으므로, 실제로는 이보다 더 오래 걸릴 수 있습니다.

{% enddetails %}

### Node 수준을 넘어서

Node 수준을 넘어서면, GPU network의 topology는 덜 표준적입니다. NVIDIA는 단일 node보다 더 큰 GPU 집합을 연결하는 reference DGX SuperPod 아키텍처를 발표하지만, 고객과 datacenter provider는 필요에 따라 이를 자유롭게 사용자 정의할 수 있습니다.

다음은 표준 1024 GPU H100 시스템의 diagram으로, 맨 아래 행의 각 DGX pod가 8개의 GPU를 가집니다. 32개 node의 각 세트를 "Scalable Unit"(또는 SU)라고 하며, 8개의 leaf switch 아래에 있습니다. 이 SU는 node당 4개의 NVSwitch와 8개의 leaf switch를 가진 256개의 GPU를 가집니다. 전체 SuperPod는 그 다음 16개의 top level "spine" switch를 추가하여, 512개의 node-level switch, 32개의 leaf switch, 16개의 spine switch로 총 512 + 32 + 16 = 560개의 NVSwitch를 가진 1024개의 GPU를 제공합니다. Leaf switch는 32개 node의 세트로 node에 연결되므로, 256개 GPU의 각 세트는 8개의 leaf switch를 가집니다. 모든 leaf switch는 모든 spine switch에 연결됩니다.

각 수준에서 사용 가능한 NVLink bandwidth, cabling, 또는 총 switch bandwidth에 의해 병목될 수 있습니다.

  * **Node 수준:** node 수준에서, 4 * 1.6TB/s = 6.4TB/s의 단방향 switch bandwidth를 가지지만, 8개의 GPU 각각은 450GB/s만 switch로 egress할 수 있으므로, 실제로 node 내에서 450e9 * 8 = 3.6TB/s의 peak bandwidth를 가집니다.
  * **SU/leaf 수준:** SU 수준에서, 8개의 switch가 1x400 Gbps Infiniband로 32개 node를 all-to-all 방식으로 연결합니다. 이는 node에서 `8 * 32 * 400 / 8 = 12.8TB/s`의 egress bandwidth를 제공하고, switch 수준에서 8 * 1.6TB/s = 12.8TB/s를 가지므로, 둘 다 정확히 일치합니다.
  * **Spine 수준:** spine 수준에서, 16개의 switch가 2x400 Gbps link로 32개의 leaf switch를 연결하므로, 32 * 16 * 400 * 2 / 8 = 51.2TB/s의 egress bandwidth를 가집니다. 16개의 switch는 16 * 1.6TB/s = 25.6TB/s의 bandwidth를 제공하므로, 이것이 이 수준에서의 병목입니다.

GPU당, 이는 node 수준에서 450GB/s, SU 수준에서 50GB/s, spine 수준에서 25 GB/s의 GPU to GPU bandwidth를 제공합니다.

| Level     | Number of GPUs | Number of NVSwitches per Unit | Total Bandwidth per Unit (TB/s, full-duplex) | GPU-to-GPU Bandwidth (GB/s, full-duplex) |
| :--------: | :-------------: | :----------------------------: | :-------------------------------------------: | :---------------------------------------: |
| Node      | 8              | 4                             | 3.6                                          | 450                                      |
| Leaf (SU) | 256            | 8                             | 12.8                                         | 50                                       |
| Spine     | 1024           | 16                            | 25.6                                         | 25                                       |

비교하자면, TPU v5p는 link당 약 90GB/s egress bandwidth, 또는 모든 축을 따라 540GB/s egress를 가집니다. 이는 point-to-point가 아니므로 제한적이고 uniform communication pattern에만 사용할 수 있지만 bandwidth 손실 없이 8000개의 TPU까지 확장할 수 있습니다.

GPU switching fabric은 이론적으로 추가 switch 또는 indirection layer를 추가하여 임의 크기로 확장할 수 있으며, 추가 latency와 가장 먼 거리에서 bandwidth 감소의 비용을 감수합니다.

## GPU에서 Collective는 어떻게 작동하는가?

GPU는 TPU와 동일한 모든 collective를 수행할 수 있습니다: ReduceScatter, AllGather, AllReduce, AllToAll. TPU와 달리, 이들이 작동하는 방식은 node 수준에서 수행되는지 그 이상에서 수행되는지에 따라 달라집니다. 이러한 모든 collective는 NVIDIA가 NCCL("nickel"로 발음) library에서 구현하며, 실제 구현은 black-box입니다. 여기서부터는 NVSwitch tree에 대한 이론적으로 최적인 model을 논의하겠습니다.

### Node 내에서

node 수준에서 AllGather 또는 ReduceScatter의 경우, TPU처럼 ring 주위에서 각 hop에서 전체 GPU-to-GPU bandwidth를 사용하여 수행할 수 있습니다. GPU를 임의로 정렬하고 전체 GPU-to-GPU bandwidth를 사용하여 배열의 일부를 ring 주위로 보냅니다:

$$T_\text{AG or RS comms} = \frac{\text{bytes} \cdot (N - 1)}{N \cdot \text{GPU egress bandwidth}} \rightarrow \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

AllReduce의 경우, 평소와 같이 RS + AG를 두 배의 비용으로 결합할 수 있습니다. latency가 걱정된다면(예: 배열이 매우 작은 경우), node의 모든 GPU에 동시에 직접 보낼 수 있으며, 보내는 총 byte를 N(node 크기)만큼 증가시키지만 전체 operation을 한 hop에서 수행합니다.

지금까지 이는 약간 다른 전체 bandwidth를 가진 TPU와 정확히 동일합니다. 예를 들어, 450 GB/s 단방향 bandwidth를 가진 H100의 경우, AllGather(bf16[B<sub>X</sub>, F])는 대략 $T_\text{comms} = (2 \cdot B \cdot F) / 450e9$가 걸릴 것입니다.

#### In network reduction

Hopper 세대부터, NVIDIA switch는 "SHARP" (*Scalable Hierarchical Aggregation and Reduction Protocol*)를 지원하여 "in-network reduction"을 허용합니다. 이는 network switch 자체가 reduction operation을 수행하고 결과를 여러 target GPU에 multiplex하거나 "MultiCast"할 수 있음을 의미합니다. 이는 각 GPU가 데이터를 top-level switch로 보내고, 거기서 reduction을 수행하고, 각 GPU를 두 번 egress할 필요 없이 결과를 broadcast할 수 있음을 의미하므로 효과적으로 AllReduce의 비용을 절반으로 줄입니다.

$$T_\text{AR comms} = \frac{\text{bytes}}{\text{GPU egress bandwidth}}$$

참고: 이는 정확하며 $1 / N$ 인수만큼 차이나지 않는데, 각 GPU가 먼저 $B * (N - 1) / N$을 egress하고, 그 다음 local shard의 부분적으로 reduced된 버전을 받고(ingress $B / N$), reduction을 완료하고, 그 다음 $B / N$을 다시 egress하고, 그 다음 완전히 reduced된 결과를 ingress하여(ingress $B * (N - 1) / N$), 정확히 $B$ byte를 ingress하게 됩니다.

### Node 수준을 넘어서

node 수준을 넘어가면, 비용은 약간 더 미묘합니다. ring을 leaf 또는 spine switch에 대한 것으로 생각하는 일종의 ring reduction을 계속할 수 있습니다. 예를 들어, AllReduce의 경우, 먼저 node 내에서 AllReduce하고, 그 다음 leaf 내에서, 그 다음 spine 내에서, 매번 일종의 ring reduction을 수행할 수 있습니다. 이상적인 세계에서, 이들을 overlap할 수 있고 가장 느린 switch에 의해 병목되게 됩니다. 1차 근사로,

$$T_\text{comms} = \max_i\left(\frac{\text{bytes} \cdot N_{\text{subdomains}_i}}{W_i}\right)$$

여기서 $W_i$는 수준 $i$에서의 aggregate switch bandwidth입니다. 따라서 예를 들어, 위의 경우에서 8개의 subdomain을 가진 node 수준에서 3.6TB/s, 32개의 subdomain을 가진 SU 수준에서 12.8TB/s, 4개의 subdomain을 가진 spine 수준에서 25.6TB/s를 가집니다. 이는 실제로 가장 큰 비율, 즉 `max(8 / 3.6e12, 32 / 12.8e12 , 4 / 25.6e12) = max(2.2e-12, 2.5e-12, 1.56e-13)`에 의해 병목될 것임을 의미하므로, 실제로 $T_\text{comms} = B \cdot \text{2.5e-12} = B / 400e9$, 즉 최고 수준에서도 약 400GB의 AllReduce bandwidth를 가집니다.

일반적으로, AllReduce bandwidth는 $\max_i(N_{\text{subdomains}_i} / W_i)$이므로, 위에서는 leaf 수준 switch에 의해 결정되는 400GB/s입니다. AllGather는 실제 통신 볼륨이 각 수준에서 변하기 때문에 약간 더 까다롭습니다. 대략 동일하지만 $\max_i(\text{bytes} \cdot (N - 1) / W)$에 약간 더 가깝습니다.

### 실습 문제

**문제 1 [Single-node AR]:** node당 N개의 GPU를 가진 단일 node를 고려하세요. AllReduce 중에 switch에 의해 정확히 몇 byte가 ingress되고 egress되나요?

{% details 답을 보려면 클릭하세요. %}

**답:** 단계별로 해봅시다.

1.  각 GPU는 $B \cdot (N - 1) / N$ byte를 보내므로, $N \cdot B \cdot (N - 1) / N = B \cdot (N - 1)$이 ingress됩니다.
2.  부분합을 accumulate하고, 각 GPU에 $B / N$ byte를 다시 보내므로, $N \ B / N = B$ byte가 egress됩니다.
3.  residual에 대해 부분합을 로컬에서 수행한 다음, 이를 switch로 다시 보냅니다. 총 $N * B / N = B$ byte가 ingress됩니다.
4.  모든 shard를 capture하고 multicast하여, $B * (N - 1) / N$을 $N$개의 destination에 보내므로, 총 $B * (N - 1) / N * N = B * (N - 1)$이 egress됩니다.

따라서 총 $B * (N - 1) + B = B\cdot N$ byte가 ingress되고 egress됩니다. 이는 전체 throughput이 정확히 $B\cdot N / W_\text{switch}$임을 지원합니다.

{% enddetails %}

**문제 2 [SU AllGather]:** M개의 node와 node당 N개의 GPU를 가진 단일 SU만 고려하세요. AllGather 중에 node 수준 switch에 의해 정확히 몇 byte가 ingress되고 egress되나요? top-level switch는?

{% details 답을 보려면 클릭하세요. %}

**답:** 이는 위와 유사하며, 다시 단계별로 하겠습니다.

1.  각 GPU는 switch에 $B / MN$ byte를 보내므로, 총 $NB / MN = B / M$ byte ingress입니다.
2.  전체 $B / M$ byte를 spine switch로 egress합니다.
3.  spine switch에서 $B * (M - 1) / M$ byte를 ingress합니다.
4.  $B - B / MN$ byte를 $N$번 egress하므로, 총 $N * (B - B / MN) = NB - B / M$입니다.

총 $B$ ingress와 $BN$ egress이므로, egress에 의해 병목되어야 하고, 총 시간은 

$$T_\text{AllGather} = \frac{BN}{W_\text{node}} = \frac{B}{450e9}$$

spine switch의 경우, 수학이 실제로 더 간단합니다. $B / M$ byte가 M번 ingress되어야 하고(총 B byte), 그 다음 $B (M - 1) / M$이 M번 egress되어야 하므로, 총 $B * (M - 1)$가 나갑니다. 이것이 상당히 크므로, 비용은 

$$T_\text{AllGather} = \frac{B \cdot (M - 1)}{W_\text{top}}$$

{% enddetails %}

## GPU에서 LLM Scaling을 위한 핵심 사항

H100 SuperPod에 대한 위 표를 재현하면, 다음을 가집니다:

| Level | Number of GPUs | GPU-to-GPU Bandwidth (full-duplex, GB/s) | Total Bandwidth (full-duplex, TB/s) | AllReduce Bandwidth (GB/s)        |
| :---: | :------------- | :--------------------------------------- | :---------------------------------- | :-------------------------------- |
| Node  | 8              | 450                                      | 3.6 (bounded by NVLink)             | 450GB/s                           |
| Leaf  | 256            | 50                                       | 8 * 1.6 = 12.8                     | 400GB/s                           |
| Spine | 1024           | 25                                       | 16 * 1.6 = 25.6                    | 400GB/s (inherited from the leaf) |

TPU에서 했듯이 compute communication roofline을 살펴봅시다. 여기서, [이전과 같이](../training-kr), computation과 communication 시간을 비교하고 $T_\text{math} \gt T_\text{comms}$ 지점을 봅시다.

**Data parallelism:** in-network reduction이 있는 단일 node 내에서 pure data parallelism에 대해 compute-bound가 되려면, 다음을 가집니다:

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot BDF}{X \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot DF}{W_\text{AllReduce}}$$

in-network reduction이 활성화되어 있으므로 comms에서 2의 인수를 잃습니다. 따라서 compute-bound가 되려면, $2B / (XC) \gt 1 / W_\text{AllReduce}$ 또는 $B / X \gt C / (2 \cdot W_\text{AllReduce})$가 필요하므로, GPU당 batch size > `989e12 / (2 * 450e9) = 1098`만 필요하며, 이는 TPU와 매우 유사합니다(여기서 세 축 모두로 수는 850). SU 또는 spine 수준에서 이를 시도하면 $BS \gt 989e12 / (2 \cdot 400e9) = 1236$을 얻습니다.

**FSDP:** 정말 유용한 유일한 것인 FSDP의 경우, 수는 이의 두 배이므로, FSDP에 대해 GPU당 BS > 2472가 필요합니다. 이 수가 TPU보다 얼마나 큰지 주목하세요(비록 B100이 이를 2의 인수만큼 줄일 것이지만).

어떤 종류의 FSDP가 필요하므로, 이는 예를 들어 2048개의 GPU에 대해 최소 5M token의 batch size가 필요함을 의미하며, 이는 상당히 가능합니다.

**Model parallelism:** model parallelism의 경우, 이는 단일 node 내에서 $Y \< F / (898e12 \cdot (8-1) / (8 \cdot 450e9)) = F / 1746$이 필요함을 시사하므로, F=16k에 대해 9-way parallelism(또는 실제로는 8 way, 이것이 node가 얼마나 큰지이므로)까지 갈 수 있습니다. 분명히, 이보다 클 수는 없지만, cross-node bandwidth가 낮아서가 아니라 전체 bandwidth가 낮기 때문입니다.

**Mixed FSDP + model parallelism:** 어떤 형태의 model parallelism과 DP를 결합하는 것은 AllReduce의 비용을 Y(TP의 양)만큼 줄이는 TPU mesh에서만큼 간단하지 않습니다. tree $\text{AllReduce}_X(A_Y { U_X })$(Y가 inner axis라고 가정)에 대한 일반적인 규칙은 다음과 같은 것 같습니다:

$$T_\text{comms} = \max_i\left[\frac{B \cdot S_i}{\max(Y, S_{i-1}) \cdot W_i}\right]$$

여기서 $S_i$는 M * N * …, tree의 수준 $i$에서 subnode의 크기입니다. 따라서 Y가 64라면, node 수준에서 `B * 8 / (64 * 3.6e12)`, leaf 수준에서 `B * 256 / (64 * 12.8e12)`, spine 수준에서 B * 1024 / (256 * 25.6e12)를 가지거나, 다시 말해 28.8e12, 3.2e12, 6.4e12의 bandwidth를 가집니다. 이는 node 수준에서 효과적으로 64의 인수를 얻고, leaf 수준에서 bandwidth에서 8의 인수를, spine 수준에서는 일정하게 유지함을 의미합니다. 256-way expert sharding을 했다면, node 수준에서 256x speedup, leaf 수준에서 32x, spine 수준에서는 speedup이 없을 것입니다.

따라서 8-way model parallelism을 하면, 실제로 node-level reduction의 비용을 8만큼 줄이고 다른 모든 것을 동일하게 둡니다. 따라서 거의 free이지만 reduction의 전체 비용을 줄이는 데는 유용하지 않습니다.

**Expert parallelism:** Expert parallelism은 이 책의 본문에서 자세히 논의되지 않지만, 1차 근사로 MoE는 dense model의 MLP weight를 E배 복제합니다. 즉, $W_\text{in}[D, F]$와 $W_\text{out}[F, D]$를 $W_\text{in}[E, D, F]$와 $W_\text{out}[E, F, D]$로 바꾸지만, 각 token은 이 중 k개만 활성화합니다. 이는 총 memory를 E만큼 증가시키지만 FLOP을 k배만 증가시킵니다. 이는 token을 선택된 expert에게 보내는 두 개의 AllToAll을 추가하지만, DP 설정에서 E배 더 많은 byte를 AllReduce해야 합니다.

AllToAll은 직접 point-to-point로 할 수 있기 때문에 GPU에서 TPU보다 간단하므로, 이런 의미에서 이야기가 훨씬 더 깔끔합니다.

하지만 추가 memory가 더 큰 문제입니다. TPU에서는 expert를 완전히 독립적인 축을 따라 단순히 shard할 수 있습니다. 예: $W[E_z, D_X, F_Y]$, 이는 우리를 거의 표준 dense model의 2축으로 줄입니다. 하지만 GPU에서는 더 이상 expert를 별도의 축을 따라 shard할 수 없으므로, 더 이상 추가 communication의 비용을 완전히 완화할 수 없습니다. 위에서 언급했듯이, 어떤 종류의 더 많은 sharding, 특히 expert parallelism을 통해 AllReduce의 비용을 어느 정도 줄일 수 있습니다. node 수준에서 8-way expert parallelism을 했다면, node-level 비용은 줄이지만 spine 수준 비용은 줄이지 않아서, 이는 나쁩니다. 64-way 또는 심지어 256-way를 하면 상당한 승리를 얻지만, DP=X와 EP=Y에 대해, expert당 k token으로, (backward pass에서) roofline은 더 이상 그렇게 간단하지 않습니다:

$$T_\text{math} = \frac{2 \cdot 2 \cdot 2 \cdot k \cdot B \cdot D \cdot F}{X \cdot Y \cdot C}$$

$$T_\text{comms} = \frac{2 \cdot 2 \cdot E \cdot D \cdot F}{W_\text{AllReduce}}$$

여기서 $W_\text{AllReduce}$는 leaf 수준에서 8배 높을 것입니다. 이는 여전히 다음을 제공합니다:

$$\frac{B}{X} \gt \frac{E \cdot C}{2 \cdot k \cdot W_\text{AllReduce}}$$

따라서 8-way EP로는 leaf 수준에서 실제로 이익을 얻지 못합니다. 256개의 expert와 8개가 활성화된(대략) DeepSeek을 64-way EP(pipeline parallelism 무시)로 취하면, 갑자기 비용이 `1236 * 256 / 8 / 8 = 4944`로 떨어져서, `4944 * 2048 = 10M` token이 필요함을 의미하며, 이는 실제로 가능합니다. 이는 FSDP의 추가 2의 인수를 무시한 것으로, 이를 20M으로 끌어올릴 것입니다.

**Pipeline parallelism:** Pipeline parallelism은 top-level switch를 통해 작은 메시지(activation 또는 activation gradient)를 hop하는 것뿐이므로 매우 작은 비용을 가집니다. 이는 추가 형태의 model parallelism으로 계산되지만, FSDP를 더 어렵게 만듭니다.

**DeepSeek은 무엇을 하나요?** 참고로, DeepSeek은 2048개의 H800 GPU로 다음과 함께 training됩니다:

  * 8개 node에 걸친 64-way Expert Parallelism (EP) 
  * 16-way Pipeline Parallelism (PP)
  * 2-way ZeRO-1 Data Parallelism (DP)

그들은 `4096 * 15360 = 62,914,560` token의 steady state batch size를 가졌습니다. 64-way EP와 16-way PP로 총 1024-way model parallelism이 되며, 이는 AllReduce가 spine 수준에서 수행됨을 의미합니다. 이는 fat tree에서 작업할 훨씬 더 많은 bandwidth를 제공하므로, 더 많은 data parallelism에 문제가 없습니다. 4-way pipeline parallelism만큼 적게 할 수 있고 여전히 이 상황에 있을 수 있지만, 그것은 자체 문제(bubble)가 있습니다.

**GPU scaling의 TLDR:**

* Pure data parallelism은 SHARP가 AllReduce 비용을 줄이기 때문에 놀랍지만, 큰 model에는 그리 유용하지 않습니다.
* FSDP는 괜찮고, AG + RS 또는 AR + AG를 해야 하기 때문에 pure DP의 2배 비용과 roofline(두 번째가 pipelining에 더 좋음). ZeRO-1은 pipelining과 작동하고, ZeRO-3는 안 됩니다. GPU당 ~2k token 필요.
* MP + FSDP는 괜찮지만 그리 좋지 않고, model parallelism은 순수한 bandwidth 이유로(cross-node bandwidth가 적다는 것과는 관련 없음) node를 넘어 확장할 수 없습니다. B100/B200에서 더 좋습니다. GPU당 memory를 줄이지만 다른 방면으로는 도움이 되지 않습니다. 즉, leaf-level bandwidth를 줄이지 않기 때문에 critical batch size를 줄이지 않습니다.
* 일반적으로, MoE + DP는 expert가 너무 chunky하기 때문에 약간 더 어려우므로, 많은 expert parallelism을 하지 않으면 DP/FSDP roofline이 올라갑니다. 정말로 node 수준을 넘어서는 expert parallelism이 필요하며, 이상적으로는 EP + MP가 전체 SU를 차지하므로, 예를 들어 8-way model parallelism, 32-way expert parallelism이 잘 작동합니다. leaf-level bandwidth를 줄이기 위해 많은 node에 걸쳐야 합니다.
* Pipeline parallelism은 zero-bubble pipelining의 코드 복잡성을 처리할 수 있다면 괜찮게 작동합니다. ZeRO-3를 불가능하게 만들므로, 대신 ZeRO-1을 해야 합니다. Model parallelism으로 계산됩니다.
* **주요 TLDR:**
    * 작은 dense model의 경우, 8-way TP + pure data parallelism이 매우 강할 것입니다. H100의 경우 bf16에서 64B model까지 갈 수 있습니다.
    * 더 큰 dense model의 경우, batch size에 따라 TP + PP 또는 FSDP를 할 수 있습니다. 더 많은 PP는 더 작은 batch size로 갈 수 있게 하지만, 대부분의 경우 FSDP가 괜찮습니다.
    * MoE의 경우, spine 수준까지 올라가는 TP + EP + PP의 일부 조합을 할 수 있고, 그러면 fat tree로 spine 수준에서 많은 bandwidth를 가지므로 괜찮습니다. node-level TP를 넘어갈 수 없고, EP는 expert 수와 imbalance + A2A의 비용에 의해 제한되고, PP는 microbatch 크기에 의해 제한됩니다. 그 다음 그 너머에서 pure DP 또는 ZeRO-1/3을 합니다.

### 실습 문제

**문제 1 [Cross-node AR 비용]:** N개의 GPU를 가진 단일 node에 걸쳐 sharded된 배열 bf16[D<sub>X</sub>, F<sub>Y</sub>]를 고려하세요. $\text{AllReduce}(bf16[D, F_Y] { U_X })$는 얼마나 걸리나요? in-network reduction을 한다고 가정할 수 있습니다. 단일 node 이상을 가진다면 어떻게 다른지 설명하세요?

{% details 답을 보려면 클릭하세요. %}

**답:** 위의 유사한 질문에 대한 답을 수정하려고 시도할 수 있습니다. 기본적으로, 먼저 각 GPU에서 $B * (X - 1) / XY$ byte를 egress하고, 그 다음 각 GPU에 $B / XY$를 다시 보내고, 그 다음 같은 양을 switch로 다시 보내고, 그 다음 각 GPU에 $B * (X - 1) / XY$를 다시 보냅니다. 총 $NB / Y$ ingress와 egress이므로, 총 시간은 $T_\text{comms} = NB / (Y \cdot W_\text{switch}) = N \cdot 2DF / (\left Y\rvert \cdot W_\text{switch})$이므로, 총 시간은 Y와 함께 감소합니다.

단일 node를 넘어가면, 위와 대략 같은 reduction을 할 수 있지만, node-level switch를 egress할 때 B / Y만이 아니라 모든 B byte를 보내야 합니다. 이는 각 model shard를 별도로 유지해야 하기 때문입니다.

{% enddetails %}

**문제 2 [Spine level AR 비용]:** 위와 같은 설정이지만 256-way model parallelism(따라서 AR이 spine 수준에서 발생)을 고려하세요. AllReduce는 얼마나 걸리나요? 여기서 GPU당 어떤 batch size를 처리할 수 있나요?

{% details 답을 보려면 클릭하세요. %}

**답:** 이는 spine 수준에서 상당히 터무니없는 양의 bandwidth를 활용할 수 있게 합니다. 4개 node에 걸쳐 25.6TB/s의 bandwidth를 가지므로, 6.4TB/s의 AllReduce bandwidth를 가집니다. SHARP를 사용하면, 이는 최소 $2 \cdot D \cdot F / 6.4e12$초가 걸릴 수 있습니다.

이는 이론적으로 GPU당 `989e12 / (2 * 6.4e12) = 77` token만큼 작은 batch size를 가질 수 있음을 의미하거나, SU당 19,712개로, 이는 꽤 wild합니다. 이는 node 내에서 8-way model parallelism과 node 간에 32-way expert parallelism, 또는 어떤 형태의 pipelining을 한 경우일 수 있습니다.

{% enddetails %}

## B100과 GB200 NVL72에서는 어떻게 달라지는가?

Broadwell은 두 배의 전체 bandwidth(900GB/s)를 가진 NVLink 5와 훨씬 더 큰 node(NVL72에서 72개 GPU)를 포함하여 많은 주요 networking 변화를 도입합니다. 다음은 diagram입니다:

{% include figure.liquid path="assets/gpu/b100-node.png" class="img-fluid" caption="<b>그림:</b> 18개의 switch와 72개의 GPU로 B100/B200 NVL72 node가 어떻게 구성되는지 보여주는 diagram." %}

1차 효과는 모든 roofline이 약 두 배 정도 좋아진다는 것입니다: 모든 AllReduce와 AllGather가 두 배 빠르므로, 두 배를 할 수 있습니다. pure data parallelism에 대해 계산한 BS > 1098 bound는 549로 감소하며, 이는 TPU v5p에 가깝습니다. F = 16000에 대한 model parallelism bound는 18로 증가하여, 거의 두 배의 model parallelism을 할 수 있음을 의미합니다.

NVIDIA는 또한 두 layer의 switch를 가지지만 모든 GPU 간에 full bandwidth를 달성할 수 있는 576 GPU GB200 NVL576 topology를 구축할 계획을 가지고 있습니다. 이는 대략 node이지만 더 먼 GPU 간에 약간의 추가 latency를 가질 것입니다. 이는 아직 출시되지 않았습니다.

{% include figure.liquid path="assets/gpu/nvl-576.png" class="img-small" caption="<b>그림:</b> Broadwell에서 576 GPU node를 볼 수 있는 방법을 보여주는 diagram." %}