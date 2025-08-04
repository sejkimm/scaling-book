---
layout: distill
title: "결론 및 추가 읽을거리"
# permalink: /main/
description: "읽어주셔서 감사합니다! 여기에 추가 학습을 위한 몇 가지 참고자료를 포함하겠습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 11

previous_section_url: "../jax-stuff-kr"
previous_section_name: "Part 10: JAX"

next_section_url: ""
next_section_name: "..."

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
  - name: "감사의 말"
  - name: "추가 읽을거리"
  - name: "피드백"

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
  .algorithm {
    padding: 10px;
    margin-top: 5px;
    margin-bottom: 5px;
    border-style: dashed;
    background-color: #fffaf2;
  }

  .algorithm li {
    margin-bottom: 0px;
  }
---

**이 에세이 시리즈를 읽어주셔서 감사하며, 끝까지 완주하신 것을 축하드립니다.** 마무리하기 전에, 몇 가지 감사의 말을 전하겠습니다:

## 감사의 말

이 문서는 Google DeepMind의 많은 사람들의 상당한 집단적 투자를 나타내며, 간단히 감사를 표하고 싶습니다!

* James Bradbury, Reiner Pope, Blake Hechtman이 원래 이 원고의 많은 아이디어를 도출했으며, Transformer의 시스템 관점을 이해하는 데 앞장섰습니다.
* Sholto Douglas가 이 문서의 첫 번째 버전을 작성했으며 프로젝트를 시작한 책임이 있습니다. 그는 누구보다도 이 문서의 전체적인 내러티브에 책임이 있습니다.
* Jacob Austin이 이 첫 번째 버전을 거친 노트에서 더 세련되고 포괄적인 결과물로 변환하는 작업을 주도했습니다. 그는 편집, 포맷팅, 이 문서 발행의 많은 작업을 했으며, 다른 저자들의 기여를 조정했습니다.
* 대부분의 그림과 애니메이션은 Anselm Levskaya와 Charlie Chen이 만들었습니다.
* Charlie Chen이 inference 섹션을 작성하고 많은 inference 그림을 그렸습니다.
* Roy Frostig이 출판, 편집, 그리고 여정의 많은 다른 단계들을 도왔습니다.

또한 과정 전반에 걸쳐 중요한 피드백을 제공한 많은 다른 분들에게도 감사드리고 싶습니다. 특히 Zak Stone, Nikhil Sethi, Caitlin Stanton, Alex Dimitriev, Sridhar Lakshmanamurthy, Albert Magyar, Diwakar Gupta, Jeff Dean, Corry Wang, Matt Johnson, Peter Hawkins, 그리고 많은 다른 분들께 감사드립니다. HTML 포맷팅을 도와준 Ruiqi Gao에게도 감사합니다.

**모두 감사합니다!**

## 추가 읽을거리

다음을 포함하여 관련된 글들이 많이 있습니다:

* [**TPU Deep Dive**](https://henryhmko.github.io/posts/tpu/tpu.html): 이 책의 정신으로 TPU 아키텍처를 심도 있게 다룬 훌륭한 글입니다.
* [**Making Deep Learning Go Brrrr From First Principles**](https://horace.io/brrr_intro.html): LLM roofline과 성능 엔지니어링에 대한 GPU와 PyTorch에 더 초점을 맞춘 튜토리얼입니다.
* [**Writing TPU Kernels with Pallas**](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html): 점점 더 TPU 프로그래밍은 Pallas에서 custom kernel을 작성하는 것을 포함합니다. 이 시리즈는 kernel을 작성하는 방법과 여기서 언급되지 않은 많은 lower level TPU 세부사항을 논의합니다.
* [**How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog**](https://siboehm.com/articles/22/CUDA-MMM): GPU와 CUDA에 특화되어 있지만, 이는 CUDA에서 matmul kernel을 최적화하는 방법을 보여주는 훌륭한 블로그 포스트입니다. 이는 TPU와 GPU가 어떻게 다른지에 대한 좋은 심층 탐구가 될 수 있습니다.
* [**Distributed arrays and automatic parallelization**](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html): 이는 JAX의 parallelism API에 대한 정말 좋은 가이드이며 여기서 논의한 아이디어들을 실제로 구현하는 방법을 배우는 좋은 방법입니다.
* [**Rafi Witten's High Performance LLMs 2024 Class**](https://github.com/rwitten/HighPerfLLMs2024): 우리의 전 동료 Rafi가 TPU 성능 엔지니어링에 대한 훌륭한 강의를 했으며 슬라이드가 모두 GitHub에 있습니다. 이는 여기서 다루는 것보다 더 깊이 있게 많은 것들을 다룹니다.
* [**\[2211.05102\] Efficiently Scaling Transformer Inference**](https://arxiv.org/abs/2211.05102): Transformer inference의 수학에 대한 상세한 논문입니다. 이는 이 문서의 많은 부분에 대한 영감입니다.
* [**Huggingface Ultra-Scale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook): 이 책의 GPU 아날로그 같은 것으로, 이는 PyTorch가 training 중에 parallelism 기술과 memory-saving 기술을 어떻게 구현하는지에 대해 더 깊이 이야기합니다.
* [**Transformer Inference Arithmetic**](https://kipp.ly/transformer-inference-arithmetic/): 이 책과 같은 아이디어를 많이 가지고 있으며 일부 훌륭한 일러스트를 포함한 블로그입니다.
* [**Stanford CS336 Slides and Videos**](https://stanford-cs336.github.io/spring2025/index.html#coursework): LLM training과 serving의 많은 세부사항을 다루는 환상적인 Stanford 강의로, 몇 가지 유용한 연습문제가 있습니다. Assignment 1과 2가 특히 관련이 있습니다.
* [**Stas Bekman's ML Engineering Handbook**](https://github.com/stas00/ml-engineering): ML 인프라에 대한 매우 실용적인 가이드로, cloud provider와 협상하는 방법, cluster 관리, GPU throughput의 경험적 측정과 같이 이 책에서 다루지 않은 주제들을 다룹니다.
  
이 영역에서 포괄적인 글쓰기의 여지가 여전히 많이 남아있으므로, 이 원고가 더 많은 것을 격려하기를 바랍니다! 또한 이것이 연구하고 공부하기에 유익한 영역이라고 믿습니다. 많은 경우에, 많은 하드웨어 accelerator를 보유하지 않고도 할 수 있습니다.

## 피드백

이를 더욱 개선할 수 있도록 코멘트나 질문을 남겨주세요. 교신 저자인 Jacob Austin에게 jaaustin [at] google [dot] com으로 연락하거나, [GitHub](https://github.com/jax-ml/scaling-book)에서 issue, pull request, 또는 discussion을 게시하여 편집을 제안할 수 있습니다.