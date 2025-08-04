---
layout: distill
title: "JAX로 TPU 프로그래밍하기"
# permalink: /main/
description: "JAX를 사용하여 TPU를 효율적으로 프로그래밍하는 방법! 이 섹션의 많은 부분은 <a href='https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html'>여기</a>에서 가져왔습니다."
date: 2025-02-04
future: true
htmlwidgets: true
hidden: false

section_number: 10

previous_section_url: "../profiling-kr"
previous_section_name: "Part 9: Profiling"

next_section_url: ../conclusion-kr
next_section_name: "Part 11: 결론"

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
  - name: "JAX에서 Parallelism은 어떻게 작동하는가?"
  - subsections:
    - name: "jax.jit: 자동 parallelism 솔루션"
    - name: "shard_map: 프로그램에 대한 명시적 parallelism 제어"
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

## JAX에서 Parallelism은 어떻게 작동하는가?

JAX는 multi-device 프로그래밍에 대한 두 가지 사고 방식을 지원합니다:

1. **컴파일러야, 맡겨!** 컴파일러가 자동으로 배열을 분할하고 주어진 프로그램을 촉진하기 위해 어떤 communication을 추가할지 결정하도록 합니다. 이를 통해 단일 device에서 프로그램을 작성하고 변경 없이 수백 개에서 자동으로 실행할 수 있습니다.
2. **그냥 내가 의도한 대로 쓰게 해줘, 제발!** 컴파일러는 좋지만, 때로는 잘못된 일을 하고 의도하지 않은 communication을 추가합니다. 때로는 우리가 무엇을 하고 있는지 매우 명시적으로 표현하고 싶습니다.

이에 따라 JAX는 각 사고 방식에 대한 두 개의 API를 제공합니다: **jit** (`jax.jit`)와 **shard\_map** (`jax.experimental.shard_map.shard_map`).

1. `jax.jit`은 프로그램의 입력과 출력 sharding을 지정할 수 있게 하고 (`in_shardings`와 `out_shardings`를 통해) [GSPMD](https://arxiv.org/abs/2105.04663) 컴파일러를 사용하여 나머지를 추론합니다. 완벽하지는 않지만, 일반적으로 프로그램을 임의의 수의 chip으로 자동 확장하는 데 괜찮은 역할을 합니다.
2. `jax.experimental.shard_map.shard_map`은 보다 명시적인 대응책입니다. 프로그램의 device-local view를 얻고 원하는 communication을 명시적으로 작성해야 합니다. sharded 배열이 있고 각 device에서 전체가 필요한가요? `jax.lax.all_gather`를 추가하세요. device 간에 배열을 합하고 싶나요? `jax.lax.psum`(AllReduce)을 추가하세요. 프로그래밍은 더 어렵지만 원하지 않는 일을 할 가능성은 훨씬 낮습니다.

<h3 id="jax-jit-the-automatic-parallelism-solution">jax.jit: 자동 parallelism 솔루션</h3>

jax.jit은 JAX 내에서 두 가지 역할을 합니다. 이름에서 알 수 있듯이, Python에서 bytecode로(XLA/HLO/LLO를 통해) 함수를 "just-in-time" 컴파일하여 더 빠르게 실행되도록 합니다. 하지만 입력이 sharded되거나 사용자가 `in_sharding` 또는 `out_sharding`을 지정하면, XLA가 여러 device에 걸쳐 계산을 분산하고 필요에 따라 communication을 추가할 수도 있습니다. 예를 들어, 다음은 jax.jit을 사용하여 sharded matmul을 작성하는 방법입니다:

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

P = shd.PartitionSpec

# TPU v5e 2x2에서 실행. 이는 하드웨어의 두 물리적 축에 이름을 할당합니다.
mesh = jax.make_mesh(axis_shapes=(2, 2), axis_names=('X', 'Y'))

# 이는 JAX에게 모든 operation에 이 mesh를 사용하도록 알려주므로, 단순히 PartitionSpec P를 지정할 수 있습니다.
shd.set_mesh(mesh)

# 우리는 device 간에 sharded된 matrix W와 입력 activation In을 생성합니다.
In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=P('Y', None))

def matmul_square(In, W):
  return jnp.einsum('bd,df->bf', jnp.square(In), W)

# 여기서 sharded matmul 함수를 명시적으로 컴파일할 수 있습니다. 이는 모든
# 필요한 comms(예: matmul 후 AllReduce)를 추가합니다.
jit_matmul = jax.jit(matmul_square, out_shardings=P('X', None)).lower(In, W).compile()

out = jit_matmul(In, W)
```

이는 어떤 sharding과도 자동으로 실행되며 device 간에 계산을 분할합니다. **하지만 하드웨어 수준에서 실제로 무슨 일이 일어나고 있을까요?**

1. 먼저 device 간에 sharded된 In과 W를 생성합니다<d-footnote>이를 어떻게 했는지 주목하세요. 이는 특정 sharding으로 배열을 생성하는 한 가지 방법입니다(즉, 생성 함수에 device argument를 추가함으로써). 다른 방법은 `jnp.array(...)`로 배열을 정상적으로 생성한 다음 예를 들어 `jax.device_put(..., P('x', 'y'))`를 하는 것입니다. 또 다른 방법은 원하는 배열을 생성하는 함수를 작성하고, `out_shardings`가 원하는 것이 되도록 jit-compile하는 것입니다.</d-footnote>. W는 contracting dimension을 따라 2-way sharded되고, In은 4-way sharded됩니다(contracting과 output dimension 모두를 따라). 이는 W[D<sub>X</sub>, F]와 In[B<sub>X</sub>, D<sub>Y</sub>] sharding에 해당하며, 일종의 model과 data parallelism입니다.
2. 이를 로컬에서(즉, 하나의 device에서) 실행한다면, `matmul_square`는 단순히 입력을 제곱하고 간단한 matmul을 수행할 것입니다. 하지만 `out_shardings`를 `P('X', None)`으로 지정했기 때문에, 출력은 batch를 따라 sharded되지만 model dimension에서는 복제되며 계산하기 위해 AllReduce가 필요합니다.

이전 섹션의 표기법을 사용하면, 이는 다음과 같은 일을 할 것입니다:

1. Out[B<sub>X</sub>, F] { U<sub>Y</sub> } = In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] 
2. Out[B<sub>X</sub>, F] = **AllReduce**(Out[B<sub>X</sub>, F] { U<sub>Y</sub> })

`jax.jit`이 이를 자동으로 추가해줍니다! 실제로 `jit_matmul.as_text()`로 HLO를 출력하고 다음 HLO를 볼 수 있습니다(대폭 축약됨):

```py
# 이 fusion은 sharded 입력과 matrix의 실제 matmul입니다
%fusion = bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} fusion(bf16[4,1024]{1,0:T(4,128)(2,1)} %param, bf16[8192,1024]{1,0:T(8,128)(2,1)S(1)} %copy-done)

# device 간에 부분적으로 합산된 결과를 reduce합니다
ROOT %AllReduce = bf16[4,8192]{1,0:T(4,128)(2,1)} AllReduce(bf16[4,8192]{1,0:T(4,128)(2,1)S(1)} %fusion)
```

위에서 matmul(fusion)과 AllReduce를 볼 수 있습니다. shape에 특히 주목하세요. `bf16[4, 1024]`는 activation의 local view입니다. `batch_size=8`이 2개 device에 분할되고 `d_model=2048`도 마찬가지로 2-way 분할되기 때문입니다.

**이는 꽤 마법적입니다!** 프로그램이 아무리 복잡해도, GSPMD와 jit은 모든 중간 activation에 대한 sharding을 찾고 필요에 따라 communication을 추가하려고 시도합니다. 그렇긴 하지만, GSPMD에는 결함이 있습니다. 실수를 할 수 있습니다. 때로는 profile을 보고 뭔가 잘못되었다는 것을 알아차릴 것입니다. 거대한 AllGather가 profile의 80%를 차지하는데, 필요 없습니다. 이런 일이 발생하면, `jax.lax.with_sharding_constraint`로 중간 tensor에 명시적으로 annotation을 달아 컴파일러를 수정하려고 시도할 수 있습니다. 예를 들어, 두 개의 matmul에서 다음과 같이 중간 activation을 `y` dimension을 따라 sharded되도록 강제할 수 있습니다(이것이 좋은 아이디어는 아니지만):

```py
import jax
import jax.numpy as jnp

mesh = jax.make_mesh((4, 2), ('X', 'Y'))

def matmul(x, Win, Wout):
  hidden = jnp.einsum('bd,df->bf', x, Win)
  hidden = jax.lax.with_sharding_constraint(hidden, jax.sharding.NamedSharding(mesh, P('x', 'y')))
  return jnp.einsum('bf,df->bd', hidden, Wout)
```

이는 컴파일러에 개입할 수 있는 유일한 방법이기 때문에 jit 세계에서 JAX parallel 프로그래밍의 약 60%를 차지합니다. Colab에서 `with_sharding_constraint`를 가지고 놀면서 어떻게 작동하는지 감을 잡아보는 것이 좋습니다. `jax.jit`를 사용하여 LLM을 작성할 때, sharding을 제어하기 위해 하는 일의 90%는 입력과 출력 sharding을 변경하고(`in_shardings`와 `out_shardings`를 통해) 올바른 comms가 발생하도록 중간 tensor에 `with_sharding_constraint`로 annotation을 다는 것입니다. 더 많은 jax.jit 예제를 보려면, [이 문서를 읽어보는 것이 좋습니다](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

<h3 id="shard-map-explicit-parallelism-control-over-a-program">shard_map: 프로그램에 대한 명시적 parallelism 제어</h3>

GSPMD가 "컴파일러가 조종하는" 모드라면, jax [shard_map](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)은 모든 것을 당신의 손에 맡깁니다. jax.jit에서처럼 입력의 sharding을 지정하지만, 모든 communication을 명시적으로 작성합니다. `jax.jit`가 프로그램의 global cross-device view를 제공하는 반면, `shard_map`은 local per-device view를 제공합니다.

다음은 예시입니다. 이 함수가 무엇을 하는지 추론해보세요:<d-footnote>mesh를 emulate하여 colab에서 직접 가지고 놀고 싶다면, 다음 cell을 사용할 수 있습니다: `import os; os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'`</d-footnote>

```py
import jax
import jax.numpy as jnp
import jax.sharding as shd

from jax.experimental.shard_map import shard_map as shmap

P = shd.PartitionSpec
shd.set_mesh(jax.make_mesh(axis_shapes=(2, 4), axis_names=('x','y')))

x = jnp.arange(0, 512, dtype=jnp.int32, device=P(('x', 'y')))

# 이 함수는 배열의 1/8에서 작동합니다.
def slice_and_average(x):
  assert x.shape == (512 // 8,)
  return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

out = shmap(slice_and_average, shd.get_abstract_mesh(), in_specs=P(('x', 'y')), out_specs=P(None,))(x)
assert out.shape == (4,)
```

**이것은 무엇을 하나요?** `slice_and_average`는 배열의 1/8과 함께 각 TPU에서 실행되며, 처음 4개 요소를 slicing하고 전체 mesh에서 평균을 냅니다. 이는 효과적으로 `mean(x[:4], x[64:68], x[128:132], …)`를 하는 것입니다. 이는 JAX에서 다른 방법으로는 표현하기 쉽지 않은 operation이기 때문에 꽤 멋집니다.

**jax.jit 대신 이렇게 하는 이유는?** `jax.jit`를 사용했다면, `slice_and_average`는 배열의 global view(전체 `[512,]` 배열)를 봤을 것입니다. 이 non-uniform slice를 slicing하고 XLA가 올바르게 해석해야 할 평균을 수행해야 했을 것입니다. XLA가 잘못된 communication을 추가하거나 혼란스러워했을 수도 있습니다. 여기서는 local view를 보고 필요한 communication만 작성합니다.

**예시 [Collective Matmul]:** 더 현실적인 예시를 들어보자면, activation이 처음에 model sharded된 model parallelism을 구현하고 싶다고 하자, 즉 A[B<sub>X</sub>, D<sub>Y</sub>] \* W[D, F<sub>Y</sub>] -> Out[B<sub>X</sub>, F<sub>Y</sub>]. 단순하게는, 먼저 A를 AllGather한 다음 local matrix multiplication을 하면 됩니다:

1. A[B<sub>X</sub>, D] = **AllGather**<sub>Y</sub>(A[B<sub>X</sub>, D<sub>Y</sub>]) 
2. Out[B<sub>X</sub>, F<sub>Y</sub>] = A[B<sub>X</sub>, D] *<sub>D</sub> W[D, F<sub>Y</sub>]

안타깝게도, 이는 communication과 computation을 overlap할 수 없기 때문에 좋지 않습니다. 이들을 overlap하는 것은 [Wang et al. 2023](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)에서 설명된 "collective matmul"로 할 수 있습니다. 알고리즘은 기본적으로 다음과 같습니다:

* 각 Y shard에 대해, A의 local chunk와 W의 local chunk의 matmul을 수행하여 `[B / X, F / Y]` shape의 결과를 생성합니다. 동시에, 다음 chunk를 로컬에서 얻도록 A를 permute하고, matmul을 수행하고, 결과를 합합니다.

shard\_map으로 이를 꽤 쉽게 구현할 수 있습니다:

```py
import functools

import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np

from jax.experimental.shard_map import shard_map

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'))
def P(*args):
  return shd.NamedSharding(mesh, shd.PartitionSpec(*args))

B, D, F = 1024, 2048, 8192
A = jnp.arange(np.prod((B, D))).reshape((B, D))
W = jnp.arange(np.prod((D, F))).reshape((D, F))

A = jax.device_put(A, P('X', 'Y'))
W = jax.device_put(W, P(None, 'Y'))

@functools.partial(jax.jit, out_shardings=P('X', 'Y'))
def matmul(lhs, rhs):
  return lhs @ rhs

def collective_matmul_allgather_lhs_contracting(lhs, rhs):
  # lhs는 looped operand; rhs는 local operand
  axis_size = jax.lax.psum(1, axis_name='Y')  # 이 예시에서 axis_size = 4
  idx = jax.lax.axis_index('Y')

  chunk_size = lhs.shape[1]
  assert rhs.shape[0] % chunk_size == 0

  def f(i, carrys):
    accum, lhs = carrys
    rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    # chunk에 대한 Matmul
    update = lhs @ rhs_chunk
    # 왼쪽으로 circular shift
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='Y',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    return accum + update, lhs

  accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
  accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs), unroll=True)

  # lhs를 찾은 상태로 두기 위해 최종 permute 후 마지막 chunk 계산
  i = axis_size - 1
  rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, (idx + i) % axis_size * chunk_size, chunk_size)
  update = lhs @ rhs_chunk
  return accum + update

jit_sharded_f = jax.jit(shard_map(
  collective_matmul_allgather_lhs_contracting, mesh,
  in_specs=(shd.PartitionSpec('X', 'Y'), shd.PartitionSpec(None, 'Y')), out_specs=shd.PartitionSpec('X', 'Y')))

shmapped_out = jit_sharded_f(A, W)
expected_out = matmul(A, W)

np.testing.assert_array_equal(shmapped_out, expected_out)
```

이는 꽤 멋집니다! 이를 benchmark해보면 훨씬 빠르다는 것을 알 수 있습니다! [여기](https://imgur.com/a/e9I6SrM)는 처음에 큰 blocking AllGather가 있는 311us가 걸리는 기본 jit matmul의 profile입니다:

{% include figure.liquid path="assets/img/not-overlapped.png" class="img-fluid" %}

그리고 [여기](https://imgur.com/a/21iy0Sv)는 244us가 걸리는 위의 버전입니다. profile에 AllGather가 없다는 것을 볼 수 있습니다. 모두 유용한 작업입니다! FLOP 활용도도 훨씬 높습니다.

{% include figure.liquid path="assets/img/overlapped.png" class="img-fluid" %}

contracting dimension에 sharding이 없는 matmul 시간은 [224us](https://imgur.com/a/i3gNKfq)라는 점도 주목할 가치가 있으므로, 여기서는 unsharded baseline에 놀랍도록 가깝습니다. 이는 TPU 활용도를 개선하기 위해 하게 될 성능 엔지니어링의 좋은 예시입니다. 더 많은 `shard_map` 예제를 보려면, [이 note가 훌륭합니다](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html#example-1-all-gather-on-one-side).

이제 `jax.jit` 또는 `shard_map`을 사용하여 구현해보려는 몇 가지 유용한 실습 문제들이 있습니다!

## 실습 문제

다음은 몇 가지 무작위 JAX 관련 문제입니다. 나중에 더 추가하겠습니다. 이 모든 것들에 대해, Colab에서 몇 개의 TPU가 필요합니다. TPUv2-8이 있는 public Colab을 사용할 수 있습니다. 이제부터 N개의 device가 사용 가능하다고 가정하겠습니다.

**문제 1:** **A**를 `X * Y = N`인 float32[S<sub>X</sub>, D<sub>Y</sub>] shape의 activation 배열이라고 하세요. 다음을 하세요:

1. 각 `(X, Y)` shard 내에서 평균을 계산하는 JAX 함수를 작성하세요. 즉, `arr[i, j]`가 shard `(i, j)`에 대한 평균인 [X, Y] 크기의 배열을 반환합니다. `jax.jit`와 `shard_map` 모두로 하세요. 각각을 profile하고 얼마나 걸렸는지 보세요. communication이 추가되었나요? *힌트: 없어야 하지만, 때로는 XLA가 어쨌든 추가합니다.*

2. **각 shard X 내에서** roll(x, shift, axis=0) - x를 어떤 shift에 대해 반환하는 JAX 함수를 작성하세요. jax.jit에서 이를 하도록 하기에는 충분히 masochist가 아니므로, `shard_map`으로만 하세요.

{% details 답을 보려면 클릭하세요. %}

Part 1: 다음은 part 1의 솔루션입니다. `jax.jit` 솔루션에 대해 해야 하는 상당히 복잡한 reshape에 주목하세요.
 
```py
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import shard_map

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

average_shmap = shard_map.shard_map(
    lambda x: x.mean(keepdims=True), 
    mesh=mesh, 
    in_specs=P('X','Y'), out_specs=P('X','Y')
)

def average(x):
  X, Y = mesh.axis_sizes
  return x.reshape(X, x.shape[0] // X, Y, x.shape[1] // Y).mean(axis=(1, 3))

average_jit = jax.jit(average, out_shardings=jax.NamedSharding(mesh, P('X','Y')))

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = average_shmap(x)
y2 = average_jit(x)

np.testing.assert_array_equal(y1, y2)
```

Part 2: 다음은 Part 2의 유사한 솔루션입니다.

```py
import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import shard_map

import functools

P = jax.sharding.PartitionSpec

mesh = jax.make_mesh((4, 2), ('X','Y'))

def shift_shmap(x, shift: int):
  shmapped = shard_map.shard_map(
      lambda x: jnp.roll(x, shift, axis=0), 
      mesh=mesh, 
      in_specs=P('X','Y'), out_specs=P('X','Y')
  )
  return shmapped(x)

@functools.partial(jax.jit, static_argnames=['shift'], out_shardings=jax.NamedSharding(mesh, P('X','Y')))
def shift_jit(x, shift: int):
  X, Y = mesh.axis_sizes
  reshaped = x.reshape(X, x.shape[0] // X, -1)
  return jnp.roll(reshaped, shift, axis=1).reshape(x.shape[0], x.shape[1])

x = jnp.arange(8 * 64 * 8, dtype=jnp.int32).reshape(8 * 64, 8)
x = jax.device_put(x, jax.NamedSharding(mesh, P('X','Y')))

y1 = shift_shmap(x, 5)
y2 = shift_jit(x, 5)

np.testing.assert_array_equal(y1, y2)
```

{% enddetails %}

**문제 2:** 여기서 기본적인 "mixture of experts" 모델을 함께 만들어보겠습니다. **W**: float32[E<sub>X</sub>, D, F<sub>Y</sub>]를 E개의 "expert" matrix 집합이라고 하세요. **A**: float32[S<sub>X</sub>, D<sub>Y</sub>](우리의 activation)라고 하고 **B**를 B[i]가 해당 activation을 처리하고 싶은 matrix를 알려주는 `[0, E)` 범위의 정수인 "routing assignment" 집합이라고 하세요. `Out[i] = W[B[i]] @ A[i]`를 반환하는 JAX 함수를 작성하고 싶습니다.

1. sharding을 완전히 무시하는 것부터 시작해봅시다. 이 모든 tensor를 하나의 device에 맞을 정도로 작게 만드세요. 이 함수의 local 구현을 작성하세요. *`[S, D, F]` shape의 배열을 materialize하지 마세요! 힌트: masking에 주의하면서 token을 `[E, S, D]` shape의 새 buffer로 sorting해보세요(두 번째 dimension이 크기 S를 가져야 하는 이유는?)*

2. 위의 방법을 단순히 `jax.jit`하면, 무언가 일어날 것입니다. 이를 profile하고 어떤 communication을 하기로 결정했는지 보세요. 얼마나 걸리나요?

3. 위에서 알아차릴 한 가지 문제는 전체 activation 집합 **A**를 local로 gather할 가능성이 있다는 것입니다. 즉, AllGather<sub>X</sub>([S<sub>X</sub>, D<sub>Y</sub>]). 이는 communication 측면에서 비싸기만 한 것이 아니라, 전체 activation 집합을 local에 맞출 수 없다면 memory 측면에서도 엄청나게 비쌉니다. `shard_map`와 명시적 communication을 사용하여 위를 구현하세요.

      1. 첫 번째 pass의 경우, `jax.lax.all_gather`를 사용하고 (a)에서처럼 reorder하는 것이 가장 쉬울 수 있습니다.

      2. 두 번째 pass의 경우, `[E, S, D]` 크기의 배열을 materialize하지 않으려고 시도해보세요. 즉, `jax.lax.while_loop` 내부의 `jax.lax.all_to_all`을 사용하여 ragged fashion으로 계산을 수행해보세요. 이렇게 하면, 전체 activation을 materialize하고 padding에 대한 compute를 낭비하는 것을 피할 수 있습니다. 이것이 원래 구현보다 얼마나 빠른가요?

4. 대부분의 MoE는 여러 (k) expert로 route하고 결과를 평균냅니다. 이를 구현하도록 위를 refactor하세요. 이 경우 k expert로 route할 **B**: int32[S, k]라고 하세요.

**문제 3:** 위의 collective matmul 예시는 실제로 실제 LLM에 매우 관련이 있습니다. 전체 Transformer stack을 하도록 예시를 수정해봅시다.

1. 연습으로, AllReduce collective matmul을 구현하는 것부터 시작해봅시다. 즉, A[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W[D<sub>Y</sub>, F] -> Out[B<sub>X</sub>, F]. 출력이 복제되지 않는다는 점에 주목하세요. naive 알고리즘은 위에서 논의되었으며, 기본적으로 local matmul 다음에 AllReduce입니다. 이 operation의 comms overlapped "collective" 버전을 만들어보세요. *힌트: output dimension에서 tile하고 `jax.lax.psum`(aka AllReduce)를 자유롭게 사용하세요.* *참고: XLA가 이를 처리하는 방식 때문에, 실제로는 baseline보다 빠르지 않을 수 있습니다.*

2. 위의 AllReduce collective matmul의 보완은 Tmp[B<sub>X</sub>, F<sub>Y</sub>] \*<sub>F</sub> W2[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]에서와 같은 ReduceScatter collective matmul입니다. 이는 Transformer의 down-projection matrix에서 발생합니다. JAX에서 이의 collective, overlapped 버전을 구현하세요. 필요한 최소한의 데이터만 전달하도록 주의하세요. *힌트: accumulate하면서 결과를 permute해보세요.*

3. 이 둘을 In[B<sub>X</sub>, D<sub>Y</sub>] \*<sub>D</sub> W<sub>in</sub>[D, F<sub>Y</sub>] \*<sub>F</sub> W<sub>out</sub>[F<sub>Y</sub>, D] -> Out[B<sub>X</sub>, D<sub>Y</sub>]를 overlapped communication으로 수행하는 end-to-end Transformer block으로 합치세요<d-footnote>이전과 마찬가지로, 여기서 생략한 non-linearity 때문에 $W_{in} \cdot W_{out}$를 먼저 할 수 없습니다.</d-footnote>. 이것이 `jax.jit` 구현보다 얼마나 빠른가요?

**문제 4:** 위에서 구현한 모든 collective matmul은 단방향입니다: 한 방향으로만 permute합니다. collective AllReduce matmul과 collective ReduceScatter matmul을 양방향 communication을 사용하도록 다시 작성하세요. 이들이 얼마나 빠른가요?

### Part 10은 여기까지입니다. 기본적으로 그게 다입니다! 최종 결론과 추가 읽을거리를 보려면 [여기](../conclusion-kr)를 클릭하세요.